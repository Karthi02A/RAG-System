import os
import io
import uuid
import logging
import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import StreamingResponse
from pymongo.errors import ServerSelectionTimeoutError

from database import get_database
import ml_engine
from data_analysis import analyze_dataframe

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

# Directory for persistent dataset storage
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BACKEND_DIR, "data_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Temporary in-memory cache for datasets
DATASET_CACHE: dict[str, pd.DataFrame] = {}


def _get_storage_path(dataset_id: str) -> str:
    return os.path.join(STORAGE_DIR, f"{dataset_id}.csv")


def _load_dataset(dataset_id: str) -> pd.DataFrame:
    """ Load dataset from cache or disk. """
    if dataset_id in DATASET_CACHE:
        return DATASET_CACHE[dataset_id]
    
    path = _get_storage_path(dataset_id)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            DATASET_CACHE[dataset_id] = df
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id} from disk: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read dataset from disk: {str(e)}")
    
    raise HTTPException(
        status_code=404, 
        detail=f"Dataset '{dataset_id}' not found. Please upload the file again."
    )


def _make_serializable(obj):
    """
    Recursively convert an object to JSON-serializable Python native types.
    Handles: datetime, numpy scalars, nested dicts, lists, and NaN/Inf.
    """
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, int, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, float, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None 
        return val
    elif isinstance(obj, (np.ndarray,)):
        return _make_serializable(obj.tolist())
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    
    # Robust Encoding Detection
    decoded_content = None
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            decoded_content = content.decode(encoding)
            break 
        except UnicodeDecodeError:
            continue
    
    if decoded_content is None:
        decoded_content = content.decode("utf-8", errors="replace")

    try:
        df = pd.read_csv(io.StringIO(decoded_content))
    except Exception as e:
        logger.error(f"pd.read_csv failed for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    dataset_id = str(uuid.uuid4())
    DATASET_CACHE[dataset_id] = df
    
    # Save to disk
    try:
        storage_path = _get_storage_path(dataset_id)
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write(decoded_content)
    except Exception as e:
        logger.error(f"Disk storage failed for {dataset_id}: {e}")

    dataset_meta = {
        "dataset_id": dataset_id,
        "dataset_name": file.filename,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "rows_count": int(df.shape[0]),
        "cols_count": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
    }

    # Non-fatal DB write
    try:
        db = get_database()
        if db is not None:
            doc_to_insert = dict(dataset_meta)
            doc_to_insert["upload_date"] = datetime.now(timezone.utc)
            await db["Datasets"].insert_one(doc_to_insert)
    except Exception:
        pass

    return _make_serializable({
        "message": "Dataset uploaded successfully",
        "dataset_id": dataset_id,
        "dataset_metadata": dataset_meta,
    })


@router.post("/analyze_dataset")
async def analyze_dataset_api(dataset_id: str):
    df = _load_dataset(dataset_id)

    try:
        analysis_results = analyze_dataframe(df)
    except Exception as e:
        logger.error(f"Analysis Failed: {e}", exc_info=True)
        # Catch and re-raise as 400 if it's a known data issue
        if isinstance(e, ValueError):
            raise HTTPException(status_code=400, detail=str(e))
        raise

    # Non-fatal DB write
    try:
        db = get_database()
        if db is not None:
            result_doc = {
                "dataset_id": dataset_id,
                "analysis_date": datetime.now(timezone.utc),
                **analysis_results,
            }
            await db["AnalysisResults"].insert_one(result_doc)
    except Exception:
        pass

    return _make_serializable({
        "dataset_id": dataset_id,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        **analysis_results,
    })


@router.post("/train_model")
async def train_model_api(dataset_id: str, target_column: str):
    df = _load_dataset(dataset_id)

    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found.")

    try:
        # Pass dataset_id for model filename persistence
        results = ml_engine.train_and_evaluate_models(df, target_column, dataset_id=dataset_id)
        return _make_serializable(results)
    except ValueError as ve:
        # CATCH INSUFFICIENT DATA ERRORS AND RETURN 400
        logger.warning(f"ML Logic Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


@router.post("/clean_dataset")
async def clean_dataset_api(dataset_id: str):
    df = _load_dataset(dataset_id)
    
    try:
        clean_df = ml_engine.auto_clean_dataset(df)
        # Generate a new ID for the clean version to not overwrite raw data
        clean_id = f"clean_{dataset_id}"
        DATASET_CACHE[clean_id] = clean_df
        
        # Save clean version to disk
        path = _get_storage_path(clean_id)
        clean_df.to_csv(path, index=False)
        
        # Recalculate health score for the cleaned version
        analysis_results = analyze_dataframe(clean_df)
        
        return {
            "message": "Dataset cleaned successfully.",
            "dataset_id": clean_id,
            "original_rows": len(df),
            "new_rows": len(clean_df),
            "removed_duplicates": len(df) - len(clean_df),
            "new_health_score": analysis_results['health_score'],
            "analysis": analysis_results
        }
    except Exception as e:
        logger.error(f"Cleaning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_api(dataset_id: str, target_column: str, inputs: dict):
    """
    v12 Prediction API: Endpoint to get real-time predictions from a forged model.
    """
    try:
        results = ml_engine.predict_from_model(dataset_id, target_column, inputs)
        return _make_serializable(results)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/validate_target")
async def validate_target_api(dataset_id: str, target_column: str):
    df = _load_dataset(dataset_id)
    try:
        results = ml_engine.validate_target_column(df, target_column)
        return _make_serializable(results)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml_eligible_columns")
async def ml_eligible_columns_api(dataset_id: str):
    df = _load_dataset(dataset_id)
    try:
        eligible_cols, _ = ml_engine.get_eligible_columns(df)
        return {"eligible_columns": eligible_cols}
    except Exception as e:
        logger.error(f"Eligibility check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml_readiness")
async def ml_readiness_api(dataset_id: str):
    df = _load_dataset(dataset_id)
    try:
        readiness = ml_engine.calculate_ml_readiness(df)
        return _make_serializable(readiness)
    except Exception as e:
        logger.error(f"Readiness calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download_model")
async def download_model_api(filename: str):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    model_path = os.path.join(model_dir, filename)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
        
    from fastapi.responses import FileResponse
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=filename
    )

@router.get("/download_clean")
async def download_clean_api(dataset_id: str):
    df = _load_dataset(dataset_id)
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cleaned_{dataset_id}.csv"}
    )
