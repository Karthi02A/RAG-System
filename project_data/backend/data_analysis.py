import pandas as pd
import numpy as np

def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Performs strictly dynamic, professional-grade data analysis (v10).
    Implements 12-point pipeline with enhanced correlation and suggestion logic.
    """
    if df.empty:
        return {
            "health_score": 0.0,
            "total_missing": 0,
            "duplicates": 0,
            "total_outliers": 0,
            "columns_count": 0,
            "rows_count": 0,
            "dataset_size_kb": 0
        }

    rows, cols = df.shape
    all_cols = df.columns.tolist()

    # --- DATASET METRICS ---
    dataset_size_bytes = df.memory_usage(deep=True).sum()
    dataset_size_kb = dataset_size_bytes / 1024
    
    rows_duplicates = int(df.duplicated().sum())
    
    duplicate_columns = []
    if cols > 1:
        try:
            duped_cols_mask = df.T.duplicated()
            duplicate_columns = df.columns[duped_cols_mask].tolist()
        except: pass

    # --- NUMERIC CONVERSION PREPROCESSING ---
    numeric_potential = {}
    for col in all_cols:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            numeric_potential[col] = {"ratio": 0.0, "series": None}
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_potential[col] = {"ratio": 1.0, "series": series}
        else:
            try:
                num_series = pd.to_numeric(series, errors="coerce")
                ratio = num_series.notna().sum() / non_null.size
                numeric_potential[col] = {"ratio": ratio, "series": num_series}
            except:
                numeric_potential[col] = {"ratio": 0.0, "series": None}

    # --- COLUMN DETECTION & CLASSIFICATION ---
    id_columns = []
    date_columns = []
    numeric_only = []
    mixed_type_cols = []
    high_cardinality_columns = []
    categorical_only = []
    classification = {}

    for col in all_cols:
        series = df[col]
        nunique = series.nunique()
        unique_ratio = nunique / rows if rows > 0 else 0
        pot = numeric_potential[col]
        
        # ID Detection
        if unique_ratio > 0.95 and pot["ratio"] > 0.95:
            num_vals = pot["series"].dropna()
            if not num_vals.empty and (num_vals % 1 == 0).all():
                name_lower = col.lower()
                id_keywords = ["id", "uuid", "key", "number", "code", "ref"]
                if any(k in name_lower for k in id_keywords):
                    id_columns.append(col)
                    classification[col] = "ID"
                    continue

        # Date Detection
        if not pd.api.types.is_numeric_dtype(series):
            try:
                date_series = pd.to_datetime(series, errors="coerce")
                if date_series.notna().mean() > 0.6:
                    date_columns.append(col)
                    classification[col] = "Date"
                    continue
            except: pass

        # Numeric Detection
        if pd.api.types.is_numeric_dtype(series) or pot["ratio"] > 0.8:
            numeric_only.append(col)
            classification[col] = "Numeric"
            continue

        # Mixed Type
        if 0.2 < pot["ratio"] < 0.8:
            mixed_type_cols.append(col)
            classification[col] = "Mixed"
            continue

        # High Cardinality
        if nunique > 50 or unique_ratio > 0.7:
            high_cardinality_columns.append(col)
            classification[col] = "High Cardinality"
            continue

        # Categorical
        categorical_only.append(col)
        classification[col] = "Categorical"

    # --- MISSING VALUES ---
    missing_counts = df.isnull().sum().to_dict()
    missing_percentages = (df.isnull().sum() / rows * 100).to_dict()
    total_missing = int(df.isnull().sum().sum())
    constant_cols = [col for col in all_cols if df[col].nunique() <= 1]

    # --- STATISTICS & ANALYSIS ---
    analysis_df = pd.DataFrame()
    for col in numeric_only:
        converted = numeric_potential[col]["series"]
        analysis_df[col] = converted if converted is not None else df[col]
    
    summary_stats = analysis_df.describe().round(2).to_dict() if not analysis_df.empty else {}

    # --- CORRELATION ANALYSIS (V10: Moderate Detection) ---
    correlation_matrix = {}
    correlation_insights = []
    total_strong_correlations = 0
    total_moderate_correlations = 0

    if len(numeric_only) >= 2:
        try:
            corr_mat = analysis_df.corr(method="pearson").round(2)
            correlation_matrix = corr_mat.to_dict()
            
            strong_pairs = []
            mod_pairs = []
            cols_list = corr_mat.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    c1, c2 = cols_list[i], cols_list[j]
                    val = corr_mat.loc[c1, c2]
                    abs_val = abs(val)
                    if abs_val >= 0.7:
                        strong_pairs.append((c1, c2, val))
                    elif abs_val >= 0.4:
                        mod_pairs.append((c1, c2, val))
            
            total_strong_correlations = len(strong_pairs)
            total_moderate_correlations = len(mod_pairs)
            
            # Use strong pairs first; fall back to moderate if none (Fix V10)
            target_pairs = strong_pairs if strong_pairs else mod_pairs
            sorted_pairs = sorted(target_pairs, key=lambda x: abs(x[2]), reverse=True)
            
            for c1, c2, val in sorted_pairs[:5]:
                label = "Strong" if abs(val) >= 0.7 else "Moderate"
                sentiment = "positive" if val > 0 else "negative"
                correlation_insights.append(f"🔗 {label} {sentiment} correlation: {c1} and {c2} ({val})")
        except: pass

    # Outlier Detection
    outliers_count = {}
    total_outliers = 0
    for col in numeric_only:
        if df[col].nunique() > 10:
            try:
                converted = numeric_potential[col]["series"]
                col_data = converted.dropna() if converted is not None else df[col].dropna()
                if not col_data.empty:
                    Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        c_outliers = int(((col_data < lower) | (col_data > upper)).sum())
                        outliers_count[col] = c_outliers
                        total_outliers += c_outliers
            except: pass

    # Insights
    insights = []
    for col in (categorical_only + high_cardinality_columns)[:3]:
        counts = df[col].value_counts()
        if not counts.empty:
            insights.append(f"Most frequent {col}: {counts.index[0]} ({ (counts.iloc[0]/rows)*100 :.1f}% of data)")

    # --- CLEANING SUGGESTIONS (V10: High-Cardinality Intelligence) ---
    suggestions = []
    missing_by_type = {}
    
    for col in all_cols:
        if df[col].isnull().any():
            ctype = classification.get(col)
            if ctype not in missing_by_type:
                missing_by_type[ctype] = []
            missing_by_type[ctype].append(col)

    # 1. High-Cardinality Special Logic
    high_card_missing = missing_by_type.get("High Cardinality", [])
    for col in high_card_missing:
        if missing_percentages[col] > 50:
            suggestions.append(f"🚨 CRITICAL: Drop high-cardinality column {col} (>50% missing)")
        else:
            suggestions.append(f"💡 Recommendation: Use Encoding for {col} instead of mode imputation")

    # 2. Standard Logic for Others
    if "Numeric" in missing_by_type:
        suggestions.append(f"Fill missing values using median for: {', '.join(missing_by_type['Numeric'])}")
    
    normal_cat_cols = missing_by_type.get("Categorical", [])
    if normal_cat_cols:
        suggestions.append(f"Fill missing values using mode (most frequent) for: {', '.join(normal_cat_cols)}")
    
    if "Date" in missing_by_type:
        suggestions.append(f"Handle missing values using Forward Fill strategy for: {', '.join(missing_by_type['Date'])}")
    
    if "Mixed" in missing_by_type:
        suggestions.append(f"Manual review recommended for Mixed-type columns: {', '.join(missing_by_type['Mixed'])}")

    if "ID" in missing_by_type:
        suggestions.append(f"Review identifier gaps for: {', '.join(missing_by_type['ID'])}")
    
    if rows_duplicates > 0:
        suggestions.append(f"Perform row deduplication ({rows_duplicates} rows)")
    if constant_cols:
        suggestions.append(f"Drop redundant single-value columns: {', '.join(constant_cols)}")
    if duplicate_columns:
        suggestions.append(f"Drop identical duplicate columns: {', '.join(duplicate_columns)}")

    # HEALTH SCORE
    m_penalty = (total_missing / (rows * cols)) * 40 if (rows * cols) > 0 else 0
    d_penalty = (rows_duplicates / rows) * 20 if rows > 0 else 0
    c_penalty = (len(constant_cols) / cols) * 20 if cols > 0 else 0
    h_score = max(0, min(100, round(100 - m_penalty - d_penalty - c_penalty, 1)))

    return {
        "health_score": h_score,
        "missing_values": missing_counts,
        "missing_percentages": {k: round(v, 2) for k, v in missing_percentages.items()},
        "total_missing": total_missing,
        "duplicates": rows_duplicates,
        "duplicate_columns": duplicate_columns,
        "outliers": outliers_count,
        "total_outliers": total_outliers,
        "constant_columns": constant_cols,
        "id_columns": id_columns,
        "date_columns": date_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "mixed_type_columns": mixed_type_cols,
        "suggestions": suggestions,
        "insights": insights,
        "summary_statistics": summary_stats,
        "correlation_matrix": correlation_matrix,
        "correlation_insights": correlation_insights,
        "total_strong_correlations": total_strong_correlations,
        "total_moderate_correlations": total_moderate_correlations, # New Key
        "columns_count": int(cols),
        "rows_count": int(rows),
        "dataset_size_kb": round(float(dataset_size_kb), 2),
        "structure": {
            "numeric": numeric_only,
            "categorical": categorical_only,
            "date": date_columns,
            "id": id_columns,
            "high_cardinality": high_cardinality_columns,
            "mixed": mixed_type_cols
        }
    }
