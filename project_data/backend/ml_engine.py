import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

logger = logging.getLogger("uvicorn.error")

# Directory for saved models
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

def calculate_ml_readiness(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"score": 0, "status": "Critical", "reasons": ["Empty dataset"]}
    total_cells = df.size
    total_missing = df.isna().sum().sum()
    missing_ratio = total_missing / total_cells if total_cells > 0 else 1
    missing_score = max(0, 100 - (missing_ratio * 200))
    num_cols = df.select_dtypes(include=[np.number]).columns
    variance_score = max(0, 100 - (len([c for c in num_cols if df[c].nunique() <= 1]) / len(df.columns) * 100)) if len(df.columns) > 0 else 50
    size_score = min(100, (len(df) / 50) * 100)
    final_score = int((missing_score * 0.4) + (variance_score * 0.3) + (size_score * 0.3))
    status = "Excellent" if final_score > 85 else "Good" if final_score > 70 else "Moderate" if final_score > 50 else "Poor"
    return {"score": final_score, "status": status, "reasons": []}

def get_eligible_columns(df: pd.DataFrame) -> tuple:
    """
    STEP 1: DATA UNDERSTANDING - Identify and Filter ML-eligible columns.
    """
    eligible_cols = []
    excluded_cols = {}
    rows_count = len(df)
    for col in df.columns:
        # 1. Drop Constant or Near-Constant Columns
        unique_count = df[col].nunique()
        if unique_count <= 1: 
            excluded_cols[col] = "Constant"
            continue
        
        # 2. Drop columns with too many missing values (> 50%)
        if df[col].isnull().mean() > 0.5: 
            excluded_cols[col] = "Missing values (>50%)"
            continue
        
        # 3. Drop Identifier Columns (unique values ≈ number of rows)
        is_object = df[col].dtype == 'object'
        is_int_seq = pd.api.types.is_integer_dtype(df[col]) and unique_count == rows_count
        
        # User requested: If a feature has unique values > 90%, remove it as an identifier
        # (Must protect continuous float features from being dropped)
        if not pd.api.types.is_float_dtype(df[col]) and unique_count > rows_count * 0.9 and rows_count > 10:
            excluded_cols[col] = "Identifier"
            continue
            
        eligible_cols.append(col)
    return eligible_cols, excluded_cols

def recommend_model(rows: int, cols: int, problem_type: str, max_relevance: float = 0) -> dict:
    """
    STEP 4: MODEL SELECTION (SMART, NOT FIXED)
    - rows < 100 -> Decision Tree
    - columns > 20 -> Random Forest
    - linear relationships (max_relevance > 0.7) -> Linear/Logistic Regression
    - Else -> Random Forest
    """
    if problem_type == "classification":
        if rows < 100:
            return {"name": "Decision Tree", "model": DecisionTreeClassifier(random_state=42, max_depth=5), 
                    "rationale": "Small dataset (< 100 rows). Decision Tree used for stability and interpretability."}
        elif max_relevance > 0.7:
            return {"name": "Logistic Regression", "model": LogisticRegression(random_state=42, max_iter=1000), 
                    "rationale": "Strong linear relationship detected. Logistic Regression provides efficient and explainable boundaries."}
        elif cols > 20:
            return {"name": "Random Forest", "model": RandomForestClassifier(random_state=42, n_estimators=100), 
                    "rationale": "High-dimensional data (> 20 features). Random Forest handles complex feature interactions effectively."}
        else:
            return {"name": "Random Forest", "model": RandomForestClassifier(random_state=42, n_estimators=100), 
                    "rationale": "Standard dataset size. Random Forest provides a balanced, robust ensemble approach."}
    else:
        if rows < 100:
            return {"name": "Decision Tree Regressor", "model": DecisionTreeRegressor(random_state=42, max_depth=5), 
                    "rationale": "Small numerical dataset. Decision Tree prevents overfitting while capturing key splits."}
        elif max_relevance > 0.7:
            return {"name": "Linear Regression", "model": LinearRegression(), 
                    "rationale": "Strong linear correlation detected. Linear Regression provides precise continuous mapping."}
        elif cols > 20:
            return {"name": "Random Forest Regressor", "model": RandomForestRegressor(random_state=42, n_estimators=100), 
                    "rationale": "Large feature set. Random Forest handles non-linear regressions without complex tuning."}
        else:
            return {"name": "Random Forest Regressor", "model": RandomForestRegressor(random_state=42, n_estimators=100), 
                    "rationale": "Balanced dataset. Random Forest offers high predictive accuracy for most shapes."}

def validate_target_column(df: pd.DataFrame, target: str) -> dict:
    """
    BASIC TARGET VALIDATION & IDENTIFIER DETECTION
    """
    if target not in df.columns: 
        return {"valid": False, "error": f"Column '{target}' not found.", "warnings": []}
    
    unique_count = df[target].nunique()
    rows_count = len(df)
    
    # 1. Constant Check
    if unique_count <= 1: 
        return {"valid": False, "error": "Target column is constant.", "warnings": []}
    
    # 2. Missing Value Check
    if df[target].isnull().mean() > 0.8:
        return {"valid": False, "error": "Target column has too many missing values (> 80%).", "warnings": []}
        
    # 3. Identifier Detection (Target Validation Edge Case)
    # If it has too many unique values relative to rows, it's likely an ID, not a learnable target.
    is_object_or_int = df[target].dtype == 'object' or pd.api.types.is_integer_dtype(df[target])
    if is_object_or_int and unique_count > rows_count * 0.9 and rows_count > 10:
        return {"valid": False, "error": "❌ The selected target appears to be an identifier and is not suitable for machine learning.", "warnings": []}
        
    return {"valid": True, "error": None, "warnings": []}

def _preprocess_data(df: pd.DataFrame, target_column: str):
    """
    DATA PREPARATION INTELLIGENCE (STEPS 1, 2, 3)
    """
    audit_log = []
    warnings = []
    
    # 1. Row Check
    data = df.copy().dropna(subset=[target_column])
    rows_count = len(data)
    if rows_count < 10: raise ValueError("Insufficient data (min 10 rows after dropping missing targets).")
    
    # 2. Target Type and Quality Detection
    y_raw = data[target_column]
    is_numeric_target = pd.api.types.is_numeric_dtype(y_raw)
    unique_target_count = y_raw.nunique()
    
    # Classification vs Regression Detection
    if not is_numeric_target or (unique_target_count < 15 and unique_target_count / rows_count < 0.2):
        problem_type = "classification"
        target_le = LabelEncoder()
        y = target_le.fit_transform(y_raw.astype(str))
    else:
        problem_type = "regression"
        y = y_raw
        target_le = None

    # 3. Feature Selection (STEP 1)
    eligible_cols, excluded_cols = get_eligible_columns(data)
    if target_column in eligible_cols: eligible_cols.remove(target_column)
    X = data[eligible_cols].copy()
    
    # Quality Diagnostics (SILENT - Do not mention data cleaning)
    missing_feat_counts = X.isnull().sum().sum()
    
    # Detect Outliers (SILENT)
    outlier_count = 0
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = X[(X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))]
        outlier_count += len(outliers)

    # Detect Scale Differences (SILENT)
    needs_scaling = False
    if len(num_cols) > 1:
        ranges = [X[c].max() - X[c].min() for c in num_cols]
        if max(ranges) / (min(ranges) + 1e-9) > 100 or max(ranges) > 1000:
            needs_scaling = True
    
    # 4. Target Validation (STEP 2: Critical Thinking)
    # Evaluate feature relevance
    max_relevance = 0
    y_series = pd.Series(y).reset_index(drop=True)
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = np.abs(pd.Series(X[col]).reset_index(drop=True).corr(y_series))
                if not np.isnan(corr): max_relevance = max(max_relevance, corr)
        except: continue
        
    weak_relationship = False
    if problem_type == "regression" and max_relevance < 0.2:
        weak_relationship = True
    elif problem_type == "classification" and max_relevance < 0.15:
        weak_relationship = True
        
    # BUILD NEW LOGGING FORMAT (Strict User Specification)
    audit_log.append(f"🎯 Target: {target_column}")
    audit_log.append(f"Type: {'Classification' if problem_type == 'classification' else 'Regression'}")
    
    if weak_relationship:
        audit_log.append("⚠️ Moderate → Limited patterns")
        warnings.append("⚠️ The selected target variable has weak relationships with input features. Model performance may be limited and predictions may not be meaningful.")
    else:
        audit_log.append("✅ Suitable for prediction")
        
    audit_log.append("")
    audit_log.append("✅ Features Used:")
    # Clean feature names for display
    clean_cols = [str(c).replace("\n", "").replace("\u200b", "").strip() for c in X.columns]
    audit_log.append(", ".join(clean_cols[:10]) + ("..." if len(clean_cols) > 10 else ""))
    
    if excluded_cols:
        audit_log.append("")
        audit_log.append("❌ Excluded:")
        for c, reason in excluded_cols.items():
            audit_log.append(f"{c} → {reason}")

    # 5. Data Preparation Intelligence (STEP 3) - SILENT
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    feat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feat_encoders[col] = le

    # Numeric Preprocessing (SILENT)
    imputer = None
    scaler = None
    if num_cols:
        imputer = SimpleImputer(strategy='median')
        X[num_cols] = imputer.fit_transform(X[num_cols])
        if needs_scaling:
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])

    metadata = {
        "problem_type": problem_type,
        "feature_cols": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "imputer": imputer,
        "scaler": scaler,
        "feat_encoders": feat_encoders,
        "target_le": target_le,
        "target_column": target_column,
        "audit_log": audit_log,
        "warnings": warnings,
        "weak_relationship": weak_relationship,
        "quality_metrics": {
            "missing": missing_feat_counts,
            "outliers": outlier_count,
            "scale_diff": needs_scaling
        }
    }
    
    return X, y, metadata

def train_and_evaluate_models(df: pd.DataFrame, target_column: str, dataset_id: str = "default") -> dict:
    """
    MAIN ORCHESTRATION PIPELINE (STEPS 5, 6, 7, 8)
    """
    # 1. Validation & Preprocessing (Steps 1, 2, 3)
    val = validate_target_column(df, target_column)
    if not val["valid"]: raise ValueError(val["error"])
    
    X, y, metadata = _preprocess_data(df, target_column)
    problem_type = metadata["problem_type"]
    rows, cols = X.shape
    
    # Calculate max relevance for model selection (linearity check)
    max_relevance = 0
    y_series = pd.Series(y).reset_index(drop=True)
    for col in X.columns:
        try:
            if pd.api.types.is_numeric_dtype(X[col]):
                corr = np.abs(pd.Series(X[col]).reset_index(drop=True).corr(y_series))
                if not np.isnan(corr): max_relevance = max(max_relevance, corr)
        except: continue

    # 2. Model Selection (Step 4)
    rec = recommend_model(rows, cols, problem_type, max_relevance)
    model = rec["model"]
    
    # 3. Training & Evaluation (Step 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    if problem_type == "classification":
        model_score = accuracy_score(y_test, test_preds)
        train_score = accuracy_score(y_train, train_preds)
        metric_name = "Accuracy"
        # Baseline: Majority Class
        majority_class_count = pd.Series(y_test).value_counts().iloc[0]
        baseline_score = majority_class_count / len(y_test)
    else:
        model_score = r2_score(y_test, test_preds)
        train_score = r2_score(y_train, train_preds)
        metric_name = "R2 Score"
        # Baseline: Mean Prediction
        mean_y = np.mean(y_test)
        baseline_preds = np.full_like(y_test, mean_y)
        baseline_score = r2_score(y_test, baseline_preds)
        # R2 baseline is 0 by definition if predicting mean, but for lift we use a small epsilon
        if baseline_score <= 0: baseline_score = 0.01 

    # Correct Model Lift formula (STEP 5)
    model_lift = ((model_score - baseline_score) / baseline_score) * 100
    
    # EDGE CASE 1: Model Lift Safety
    if model_lift > 100: model_lift = 100.0
    elif model_lift < -100: model_lift = -100.0
    
    # EDGE CASE 2: Underfitting vs Overfitting Logic
    gap = train_score - model_score
    is_underfitting = False
    
    if problem_type == "classification" and model_score <= baseline_score:
        is_underfitting = True
    elif problem_type == "regression" and model_score < 0.2:
        is_underfitting = True
        
    if is_underfitting:
        overfit_status = "Underfitting"
        overfit_desc = "The model failed to capture patterns in the data (low accuracy)."
    else:
        if gap > 0.15:
            overfit_status = "Overfitting"
            overfit_desc = "May not generalize well"
        elif 0.05 <= gap <= 0.15:
            overfit_status = "Moderate overfitting"
            overfit_desc = "May not generalize well"
        else:
            overfit_status = "Good generalization"
            overfit_desc = "Consistent performance"

    # 5. Feature Importance (Step 7) & EGDE CASE 3: Reliability Check
    importance = []
    importance_unreliable = is_underfitting or model_score <= baseline_score
    
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Support for Linear models (LinearRegression, LogisticRegression)
        if len(model.coef_.shape) > 1:
            imps = np.mean(np.abs(model.coef_), axis=0)
        else:
            imps = np.abs(model.coef_)
        # Normalize to 0-1 range to match feature_importances_ style
        total_imp = imps.sum()
        if total_imp > 0:
            imps = imps / total_imp
    else:
        imps = None

    if imps is not None:
        for i, f in enumerate(X.columns):
            val_imp = float(imps[i])
            if val_imp > 0.2: label = "High influence"
            elif val_imp > 0.05: label = "Moderate influence"
            else: label = "Low influence"
            
            note = ""
            if importance_unreliable:
                note = "⚠️ Feature importance may not be meaningful due to poor model performance."
            elif metadata["weak_relationship"]:
                note = "Importance may not be meaningful due to weak target relationship."
                
            importance.append({
                "feature": str(f).replace("\n", "").replace("\u200b", "").strip(), 
                "importance": val_imp, 
                "influence": label,
                "note": note
            })
        importance.sort(key=lambda x: x["importance"], reverse=True)

    # --- ADVANCED ML INTELLIGENCE ---
    # 1. Data Size Awareness
    if len(df) < 100:
        data_size_awareness = "Small → Limited learning"
    elif len(df) <= 1000:
        data_size_awareness = "Moderate"
    else:
        data_size_awareness = "Large → Strong learning"
        
    # 2. Prediction Difficulty
    if not metadata["weak_relationship"] and model_score > 0.85:
        prediction_difficulty = "Easy"
    elif model_score > 0.70 and not metadata["weak_relationship"]:
        prediction_difficulty = "Moderate"
    else:
        prediction_difficulty = "Hard"

    # 3. Model Confidence
    if model_score > 0.80:
        model_confidence = "High"
    elif model_score >= 0.60:
        model_confidence = "Medium"
    else:
        model_confidence = "Low"

    # 4. Model Limitation
    if gap > 0.15:
        model_limitation = "Model may not generalize well to unseen data"
    elif is_underfitting or model_score < 0.60:
        model_limitation = "Limited data or weak patterns may restrict performance"
    else:
        model_limitation = "Predictions strictly depend on historical feature patterns"

    # 5. Smart Suggestion
    if len(df) < 500:
        smart_suggestion = "Add more data to improve accuracy"
    elif gap > 0.15:
        smart_suggestion = "Reduce complex features to avoid overfitting"
    elif model_score < 0.60:
        smart_suggestion = "Try adding new informative columns (Features)"
    else:
        smart_suggestion = "Performance is solid; deploy or tune hyperparameters"

    # 6. Final Result Refinement
    if model_score > 0.80 and not metadata["weak_relationship"]:
        summary = "✅ Strong performance\n→ Predictions are reliable for most use cases"
    elif model_score < 0.60 or is_underfitting:
        summary = "❌ Low performance\n→ Do not rely on predictions"
    else:
        summary = "⚠️ Moderate performance\n→ Use for trends, not exact predictions"

    # Model Persistence
    bundle = {
        "model": model,
        "metadata": metadata,
        "metrics": {metric_name: model_score},
        "algorithm": rec["name"],
        "problem_type": problem_type
    }
    model_filename = f"model_{dataset_id}_{target_column}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(bundle, model_path)

    return {
        "problem_type": problem_type,
        "recommended_algorithm": rec["name"],
        "recommendation_rationale": rec["rationale"],
        "metrics": {metric_name: float(model_score)},
        "baseline_score": float(baseline_score),
        "model_lift": float(model_lift),
        "overfitting_diagnosis": {
            "status": overfit_status,
            "description": overfit_desc,
            "gap": float(gap)
        },
        "feature_importance": importance[:10],
        "summary": summary,
        "advanced_insights": {
            "data_size": data_size_awareness,
            "prediction_difficulty": prediction_difficulty,
            "confidence": model_confidence,
            "limitation": model_limitation,
            "suggestion": smart_suggestion
        },
        "audit_log": metadata["audit_log"],
        "warnings": metadata["warnings"],
        "model_url": f"/api/download_model?filename={model_filename}"
    }

def predict_from_model(dataset_id: str, target_column: str, input_data: dict) -> dict:
    """
    v12 Prediction Engine: Transforms raw input and predicts using the bundled model.
    """
    model_path = os.path.join(MODEL_DIR, f"model_{dataset_id}_{target_column}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")
    
    bundle = joblib.load(model_path)
    model = bundle["model"]
    meta = bundle["metadata"]
    
    # Create DF for the single row
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features exist (fill missing with 0 or Unknown)
    for col in meta["feature_cols"]:
        if col not in input_df.columns:
            input_df[col] = np.nan
    
    input_df = input_df[meta["feature_cols"]]
    
    # 1. Numeric Transformation
    if meta["num_cols"]:
        # Impute missing if any in input
        input_df[meta["num_cols"]] = meta["imputer"].transform(input_df[meta["num_cols"]])
        # Scale ONLY if scaler exists
        if meta["scaler"]:
            input_df[meta["num_cols"]] = meta["scaler"].transform(input_df[meta["num_cols"]])
    
    # 2. Categorical Transformation
    for col, le in meta["feat_encoders"].items():
        val = str(input_df.at[0, col])
        # Handle unseen labels by mapping to the first known label (or just use 0)
        try:
            input_df[col] = le.transform([val])[0]
        except:
            input_df[col] = 0 # Fallback for unseen
            
    # 3. Prediction
    pred = model.predict(input_df)[0]
    
    # 4. Inverse Transform Target (if classification)
    readable_pred = pred
    if meta["problem_type"] == "classification" and meta["target_le"]:
        readable_pred = meta["target_le"].inverse_transform([int(pred)])[0]
    
    return {
        "prediction": readable_pred,
        "is_classification": meta["problem_type"] == "classification",
        "algorithm": bundle["algorithm"]
    }

def auto_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust 100% Health Auto-Fix Pipeline (v11).
    Handles hidden duplicates, dirty numeric columns, and strict type enforcement.
    """
    clean_df = df.copy()
    
    # 1. Early Normalization: Strip Whitespace from Headers and Text
    clean_df.columns = clean_df.columns.str.strip()
    obj_cols = clean_df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        clean_df[col] = clean_df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)

    # 2. Smart Type Detection (Potential Numeric)
    all_cols = clean_df.columns.tolist()
    numeric_cols = []
    categorical_cols = []
    
    for col in all_cols:
        series = clean_df[col]
        # Treat as numeric if it already is, or if >=50% can be converted
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            try:
                converted = pd.to_numeric(series, errors='coerce')
                if converted.notna().mean() >= 0.5:
                    numeric_cols.append(col)
                    clean_df[col] = converted # Convert early
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)

    # 3. Handle Missing Values
    for col in all_cols:
        if col in numeric_cols:
            if clean_df[col].isna().any():
                median_val = clean_df[col].median()
                # Fallback if whole column is NaN
                if pd.isna(median_val): median_val = 0
                clean_df[col] = clean_df[col].fillna(median_val)
        else:
            # Categorical/Other Imputation
            if clean_df[col].isna().any():
                mode_series = clean_df[col].mode()
                if not mode_series.empty:
                    clean_df[col] = clean_df[col].fillna(mode_series[0])
                else:
                    clean_df[col] = clean_df[col].fillna("Unknown")

    # 4. Outlier Capping & Range Safety
    for col in numeric_cols:
        try:
            Q1, Q3 = clean_df[col].quantile(0.25), clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                clean_df[col] = clean_df[col].clip(lower, upper)
        except: pass

    # 5. Final Type Enforcement & Strip 'nan' strings
    for col in categorical_cols:
        clean_df[col] = clean_df[col].astype(str).replace(['nan', 'None', 'NULL'], 'Unknown')
    
    for col in numeric_cols:
        # Final pass to ensure no NaNs left (from clip or other steps)
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)

    # 6. Strict Deduplication (Rows)
    clean_df = clean_df.drop_duplicates()
    
    # 7. Redundant Column Removal (Constant & Duplicate Columns)
    # Drop constant columns (nunique <= 1)
    constant_cols = [col for col in clean_df.columns if clean_df[col].nunique() <= 1]
    if constant_cols:
        clean_df = clean_df.drop(columns=constant_cols)
        
    # Drop identical duplicate columns
    if len(clean_df.columns) > 1:
        try:
            # We use T.duplicated() to find columns that are identical
            dupe_mask = clean_df.T.duplicated()
            duplicate_columns = clean_df.columns[dupe_mask].tolist()
            if duplicate_columns:
                clean_df = clean_df.drop(columns=duplicate_columns)
        except: pass
        
    return clean_df
