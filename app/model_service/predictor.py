# predictor.py (незначительные правки)
import pandas as pd
import numpy as np
import joblib
import json
import re
import os

MODEL_DIR = "saved_model"
_loaded_artifacts_cache = None

def convert_runtime_to_minutes(value): # Без изменений
    if pd.isna(value): return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def get_artifacts(): # Без изменений в логике, но убедимся что MODEL_DIR правильный
    global _loaded_artifacts_cache
    if _loaded_artifacts_cache is not None:
        return _loaded_artifacts_cache
    print(f"Загрузка артефактов из директории: {MODEL_DIR}") # MODEL_DIR определен в этом файле
    # ... остальная часть функции get_artifacts ...
    artifacts = {}
    required_files = {
        "model": "movie_box_office_model.joblib",
        "numerical_imputer": "numerical_imputer.joblib",
        "categorical_imputer": "categorical_imputer.joblib",
        "target_encoder": "target_encoder.joblib",
        "scaler": "scaler.joblib",
        "column_info": "column_info.json"
    }
    try:
        for key, f_name in required_files.items():
            path = os.path.join(MODEL_DIR, f_name) # MODEL_DIR используется здесь
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден файл артефакта: {path}")
            
            if f_name.endswith(".joblib"):
                artifacts[key] = joblib.load(path)
            elif f_name.endswith(".json"):
                with open(path, "r", encoding='utf-8') as f:
                    artifacts[key] = json.load(f)
        
        print("Артефакты успешно загружены и кэшированы.")
        _loaded_artifacts_cache = artifacts
        return artifacts
    except Exception as e:
        print(f"Критическая ошибка при загрузке артефактов: {e}")
        _loaded_artifacts_cache = None 
        raise RuntimeError(f"Не удалось загрузить артефакты: {e}")


def preprocess_input_data(movie_data_dict, artifacts):
    column_info = artifacts['column_info']
    df_movie = pd.DataFrame([movie_data_dict])

    # 1. Feature Engineering (опыт)
    exp_maps = column_info.get("experience_maps", {})
    personnel_cols = ["director", "writer", "producer", "composer", "cinematographer"]
    for col in personnel_cols:
        map_key = f"{col}_experience_map"
        # Убедимся, что имя из фильма всегда строка для поиска в карте
        name_in_movie = str(df_movie.loc[0, col]) if col in df_movie.columns and pd.notna(df_movie.loc[0, col]) else ""
        exp_val = exp_maps.get(map_key, {}).get(name_in_movie, 0) # Если "" не найдено, будет 0
        df_movie[f"{col}_experience"] = exp_val

    actor_exp_cols_gen = []
    for i in range(1, 5):
        col = f"main_actor_{i}"
        map_key = f"{col}_experience_map"
        name_in_movie = str(df_movie.loc[0, col]) if col in df_movie.columns and pd.notna(df_movie.loc[0, col]) else ""
        exp_val = exp_maps.get(map_key, {}).get(name_in_movie, 0)
        df_movie[f"{col}_experience"] = exp_val
        actor_exp_cols_gen.append(f"{col}_experience")
    
    df_movie["cast_popularity"] = df_movie[actor_exp_cols_gen].sum(axis=1).iloc[0] if actor_exp_cols_gen else 0

    # 2. run_time
    if 'run_time' in df_movie.columns:
        df_movie['run_time'] = df_movie['run_time'].apply(convert_runtime_to_minutes)

    # 3. Numerical Imputation
    num_imputer = artifacts['numerical_imputer']
    num_features_expected = getattr(num_imputer, 'feature_names_in_', [])
    if isinstance(num_features_expected, np.ndarray): num_features_expected = num_features_expected.tolist()
    
    for col in num_features_expected: # Создаем недостающие колонки, которые ожидает импьютер
        if col not in df_movie.columns: df_movie[col] = np.nan 
    
    if num_features_expected: 
        df_movie_subset_num = df_movie[num_features_expected] 
        imputed_num_vals = num_imputer.transform(df_movie_subset_num)
        for i_num, col_name_num in enumerate(num_features_expected):
            df_movie[col_name_num] = imputed_num_vals[0, i_num]
    
    # 4. Grouped Imputation
    group_maps = column_info.get("grouped_imputation_maps", {})
    group_impute_cols = column_info.get("group_impute_director_dependent_cols", [])
    director_name_str = str(df_movie.loc[0, 'director']) if 'director' in df_movie.columns and pd.notna(df_movie.loc[0, 'director']) else ""
    if director_name_str: # Только если есть имя режиссера
        for col in group_impute_cols:
            if col not in df_movie.columns: df_movie[col] = np.nan # Создаем, если нет
            if pd.isna(df_movie.loc[0, col]): # Заполняем только если пропуск
                map_key = f"{col}_director_mode_map"
                # Ключи в group_maps (имена режиссеров) должны быть строками, как и director_name_str
                mode_val = group_maps.get(map_key, {}).get(director_name_str) 
                if mode_val is not None: df_movie.loc[0, col] = mode_val
    
    # 5. Fill 'Unknown' for specific columns
    fill_unknown_cols = column_info.get("fill_unknown_cols", [])
    for col in fill_unknown_cols:
        if col not in df_movie.columns: df_movie[col] = np.nan
        current_val = df_movie.loc[0, col]
        if pd.isna(current_val) or (isinstance(current_val, str) and current_val.strip() == ''):
            df_movie.loc[0, col] = 'Unknown'

    # 6. Categorical Imputation (general)
    cat_imputer = artifacts['categorical_imputer']
    cat_features_expected = getattr(cat_imputer, 'feature_names_in_', [])
    if isinstance(cat_features_expected, np.ndarray): cat_features_expected = cat_features_expected.tolist()

    for col in cat_features_expected: # Создаем недостающие колонки, которые ожидает импьютер
        if col not in df_movie.columns: df_movie[col] = np.nan
    
    if cat_features_expected:
        df_movie_subset_cat = df_movie[cat_features_expected]
        imputed_cat_vals = cat_imputer.transform(df_movie_subset_cat)
        for i_cat, col_name_cat in enumerate(cat_features_expected):
            df_movie[col_name_cat] = imputed_cat_vals[0, i_cat]

    # 7. One-Hot Encoding
    ohe_cols = column_info.get("one_hot_encoded_cols", [])
    actual_ohe_cols = [col for col in ohe_cols if col in df_movie.columns]
    if actual_ohe_cols:
        df_movie = pd.get_dummies(df_movie, columns=actual_ohe_cols, dummy_na=False)

    # 8. Target Encoding
    target_encoder = artifacts['target_encoder']
    te_encoder_fit_cols = getattr(target_encoder, 'cols', []) 
    cols_for_te_transform = [col for col in te_encoder_fit_cols if col in df_movie.columns]
    if target_encoder and cols_for_te_transform:
        # Перед transform убедимся, что колонки имеют правильный тип (обычно object/str для TargetEncoder)
        for col_te in cols_for_te_transform:
            if df_movie[col_te].dtype != 'object':
                 df_movie[col_te] = df_movie[col_te].astype(str) # Приводим к строке, если не строка

        transformed_te_vals = target_encoder.transform(df_movie[cols_for_te_transform])
        if isinstance(transformed_te_vals, pd.Series):
             df_movie[cols_for_te_transform[0]] = transformed_te_vals.values[0]
        elif isinstance(transformed_te_vals, pd.DataFrame):
             df_movie[cols_for_te_transform] = transformed_te_vals.values
    
    # 9. Reindex
    final_features = column_info.get("final_model_features", [])
    if not final_features: 
        raise ValueError("final_model_features не найдены в column_info.")
    
    df_processed_for_model = pd.DataFrame(0, index=[0], columns=final_features)
    cols_to_transfer = [col for col in df_movie.columns if col in df_processed_for_model.columns]
    if cols_to_transfer:
        df_processed_for_model[cols_to_transfer] = df_movie[cols_to_transfer].values

    if df_processed_for_model.isnull().any().any():
        df_processed_for_model.fillna(0, inplace=True)
    
    # 10. Scaling
    X_scaled = artifacts['scaler'].transform(df_processed_for_model)
    return X_scaled

def make_prediction(movie_data_dict): # Без изменений
    try:
        artifacts = get_artifacts() 
        if artifacts is None:
            raise RuntimeError("Не удалось загрузить артефакты модели.")
        processed_data = preprocess_input_data(movie_data_dict, artifacts)
        model = artifacts['model']
        prediction_value = model.predict(processed_data)
        predicted_gross = round(float(prediction_value[0]))
        return predicted_gross
    except Exception as e:
        print(f"Ошибка в процессе предсказания: {e}")
        import traceback
        traceback.print_exc()
        raise