# model_service/predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import xgboost as xgb

MODEL_DIR = "saved_model" 
_loaded_artifacts_cache = None

def convert_runtime_to_minutes(value):
    if pd.isna(value): return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def get_artifacts():
    global _loaded_artifacts_cache
    if _loaded_artifacts_cache is not None:
        return _loaded_artifacts_cache

    print(f"Загрузка артефактов из директории: {MODEL_DIR}")
    artifacts = {}
    required_files_map = {
        "model_json_path": "movie_box_office_model.json",
        "numerical_imputer": "numerical_imputer.joblib",
        "categorical_imputer": "categorical_imputer.joblib",
        "onehot_encoder": "onehot_encoder.joblib",
        "target_encoder": "target_encoder.joblib",
        "scaler": "scaler.joblib",
        "column_info": "column_info.json"
    }
    try:
        for key, f_name in required_files_map.items():
            path = os.path.join(MODEL_DIR, f_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден файл артефакта: {path}")
            
            if f_name.endswith(".joblib"):
                artifacts[key] = joblib.load(path)
            elif key == "column_info":
                with open(path, "r", encoding='utf-8') as f:
                    artifacts[key] = json.load(f)
            elif key == "model_json_path": 
                model_xgb = xgb.Booster()
                model_xgb.load_model(path)
                artifacts['model'] = model_xgb
                print(f"  Модель XGBoost успешно загружена из {f_name}")
        
        if 'model' not in artifacts:
             raise ValueError("Не удалось загрузить основную модель (movie_box_office_model.json).")

        print("Артефакты успешно загружены и кэшированы.")
        _loaded_artifacts_cache = artifacts
        return artifacts
    except Exception as e:
        print(f"Критическая ошибка при загрузке артефактов: {e}")
        _loaded_artifacts_cache = None 
        raise RuntimeError(f"Не удалось загрузить артефакты: {e}")


def preprocess_input_data(movie_data_dict, artifacts):
    column_info = artifacts.get('column_info')
    if not column_info:
        raise ValueError("column_info не был загружен или отсутствует в артефактах.")
        
    df_movie = pd.DataFrame([movie_data_dict])

    # Списки колонок из column_info.json
    # Используем .get для безопасного извлечения, на случай если ключ отсутствует
    cfg_numerical_cols = column_info.get("numerical_columns", [])
    cfg_categorical_cols = column_info.get("categorical_columns", [])
    
    # Удаляем целевую переменную и связанные с ней (domestic, international) из списков, если они там есть,
    # так как они не должны быть признаками для импутации или обработки во входных данных для предсказания.
    # Эти колонки (`domestic`, `international`) НЕ ДОЛЖНЫ БЫТЬ в movie_data_dict при предсказании.
    # Ваш column_info.json содержит 'domestic', 'international' в numerical_columns.
    # Это нужно исправить в ноутбуке при генерации column_info.json!
    # Для целей этого скрипта, мы их принудительно удалим из рассмотрения здесь.
    
    cols_to_exclude_from_processing = [column_info.get("target", "worldwide"), "domestic", "international"]

    cfg_numerical_cols = [col for col in cfg_numerical_cols if col not in cols_to_exclude_from_processing]
    # cfg_categorical_cols обычно не содержат target, но на всякий случай:
    # cfg_categorical_cols = [col for col in cfg_categorical_cols if col not in cols_to_exclude_from_processing]


    # 1. Feature Engineering (опыт)
    exp_maps = column_info.get("experience_maps", {}) # Предполагаем, что этот ключ есть или будет добавлен
    personnel_cols = ["director", "writer", "producer", "composer", "cinematographer"]
    for col in personnel_cols:
        map_key = f"{col}_experience_map"
        name_in_movie = str(df_movie.loc[0, col]) if col in df_movie.columns and pd.notna(df_movie.loc[0, col]) else ""
        exp_val = exp_maps.get(map_key, {}).get(name_in_movie, 0)
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
    num_imputer = artifacts.get('numerical_imputer')
    if not num_imputer: raise ValueError("numerical_imputer не найден в артефактах.")
    
    # Используем cfg_numerical_cols или feature_names_in_
    num_features_to_impute = getattr(num_imputer, 'feature_names_in_', [])
    if isinstance(num_features_to_impute, np.ndarray): num_features_to_impute = num_features_to_impute.tolist()
    if not num_features_to_impute and cfg_numerical_cols: # Если feature_names_in_ пуст, но есть инфо в JSON
        print("Предупреждение: Используется список numerical_columns из JSON для numerical_imputer.")
        num_features_to_impute = [col for col in cfg_numerical_cols if col in df_movie.columns or col in getattr(num_imputer, 'feature_names_in_', [])] # Только те, что есть или были при обучении

    # Убедимся, что все ожидаемые колонки есть в df_movie
    for col in num_features_to_impute:
        if col not in df_movie.columns: df_movie[col] = np.nan 
    
    if num_features_to_impute: 
        # Передаем только те колонки, на которых импьютер был обучен, и в правильном порядке
        # Если feature_names_in_ не пуст, он определяет порядок и набор
        cols_for_transform = num_features_to_impute 
        df_movie_subset_num = df_movie[cols_for_transform]
        imputed_num_vals = num_imputer.transform(df_movie_subset_num)
        for i_num, col_name_num in enumerate(cols_for_transform):
            df_movie[col_name_num] = imputed_num_vals[0, i_num]
    
    # 4. Grouped Imputation - логика остается прежней
    # Но нужно убедиться, что `group_impute_director_dependent_cols` и `experience_maps`
    # добавлены в ваш `column_info.json` в ноутбуке.
    group_maps = column_info.get("grouped_imputation_maps", {}) 
    group_impute_cols = column_info.get("group_impute_director_dependent_cols", []) 
    director_name_str = str(df_movie.loc[0, 'director']) if 'director' in df_movie.columns and pd.notna(df_movie.loc[0, 'director']) else ""
    if director_name_str and group_impute_cols: # Проверка, что group_impute_cols не пуст
        for col in group_impute_cols:
            if col not in df_movie.columns: df_movie[col] = np.nan
            if pd.isna(df_movie.loc[0, col]):
                map_key = f"{col}_director_mode_map"
                mode_val = group_maps.get(map_key, {}).get(director_name_str)
                if mode_val is not None: df_movie.loc[0, col] = mode_val
    
    # 5. Fill 'Unknown' for specific columns - логика остается прежней
    # `fill_unknown_cols` должен быть в column_info.json
    fill_unknown_cols = column_info.get("fill_unknown_cols", []) 
    if fill_unknown_cols: # Проверка, что не пуст
        for col in fill_unknown_cols:
            if col not in df_movie.columns: df_movie[col] = np.nan
            current_val = df_movie.loc[0, col]
            if pd.isna(current_val) or (isinstance(current_val, str) and current_val.strip() == ''):
                df_movie.loc[0, col] = 'Unknown'

    # 6. Categorical Imputation (general)
    cat_imputer = artifacts.get('categorical_imputer')
    if not cat_imputer: raise ValueError("categorical_imputer не найден в артефактах.")
    
    cat_features_to_impute = getattr(cat_imputer, 'feature_names_in_', [])
    if isinstance(cat_features_to_impute, np.ndarray): cat_features_to_impute = cat_features_to_impute.tolist()
    if not cat_features_to_impute and cfg_categorical_cols: # Если feature_names_in_ пуст
        print("Предупреждение: Используется список categorical_columns из JSON для categorical_imputer.")
        cat_features_to_impute = [col for col in cfg_categorical_cols if col in df_movie.columns or col in getattr(cat_imputer, 'feature_names_in_', [])]


    for col in cat_features_to_impute:
        if col not in df_movie.columns: df_movie[col] = np.nan
    
    if cat_features_to_impute:
        cols_for_cat_transform = cat_features_to_impute
        df_movie_subset_cat = df_movie[cols_for_cat_transform]
        imputed_cat_vals = cat_imputer.transform(df_movie_subset_cat)
        for i_cat, col_name_cat in enumerate(cols_for_cat_transform):
            df_movie[col_name_cat] = imputed_cat_vals[0, i_cat]

    # --- One-Hot Encoding с использованием sklearn.preprocessing.OneHotEncoder ---
    ohe = artifacts.get('onehot_encoder')
    if not ohe: raise ValueError("onehot_encoder не найден в артефактах.")

    # Получаем ИСХОДНЫЕ колонки, на которых OHE был обучен
    ohe_source_columns = []
    if hasattr(ohe, 'feature_names_in_'):
        ohe_source_columns = ohe.feature_names_in_.tolist()
    else: # Запасной вариант из column_info.json (менее надежный)
        # Вам нужно будет добавить ключ "one_hot_input_columns" в JSON при сохранении в ноутбуке,
        # если feature_names_in_ не сохраняется с вашим OHE.
        ohe_source_columns = column_info.get("one_hot_input_columns", []) 
        if not ohe_source_columns:
             # Если нет ни там, ни там, можно попробовать угадать из cfg_categorical_cols
             # те, которые обычно OHE-кодируются (mpaa, genre_*)
             potential_ohe_cols_from_cfg = ['mpaa', 'genre_1', 'genre_2', 'genre_3', 'genre_4']
             ohe_source_columns = [col for col in potential_ohe_cols_from_cfg if col in cfg_categorical_cols]
             if ohe_source_columns:
                 print(f"Предупреждение: Исходные колонки для OHE определены эвристически: {ohe_source_columns}")
             else:
                raise ValueError("Не удалось определить исходные колонки для OneHotEncoder.")

    # Убедимся, что все исходные колонки для OHE есть в df_movie
    for col in ohe_source_columns:
        if col not in df_movie.columns:
            # Предполагаем, что categorical_imputer уже заполнил их (например, 'Unknown')
             df_movie[col] = df_movie.get(col, pd.Series(['Unknown'], index=df_movie.index)) 
    
    if ohe_source_columns: 
        # Убедимся, что колонки имеют тип object/string для OHE
        for col in ohe_source_columns:
            if col in df_movie.columns and df_movie[col].dtype != 'object':
                df_movie[col] = df_movie[col].astype(str)

        ohe_transformed_data = ohe.transform(df_movie[ohe_source_columns])
        
        ohe_feature_names = []
        if hasattr(ohe, 'get_feature_names_out'):
            ohe_feature_names = ohe.get_feature_names_out(ohe_source_columns)
        elif hasattr(ohe, 'get_feature_names'): # Для старых версий
            ohe_feature_names = ohe.get_feature_names(ohe_source_columns)
        else:
             raise RuntimeError("Не удалось получить имена признаков из OneHotEncoder после transform.")
        
        if not isinstance(ohe_feature_names, np.ndarray) or not ohe_feature_names.size > 0 :
             raise RuntimeError("Имена признаков из OneHotEncoder некорректны или пусты.")

        if hasattr(ohe_transformed_data, "toarray"): 
            ohe_df = pd.DataFrame(ohe_transformed_data.toarray(), columns=ohe_feature_names, index=df_movie.index)
        else: 
            ohe_df = pd.DataFrame(ohe_transformed_data, columns=ohe_feature_names, index=df_movie.index)
        
        df_movie = df_movie.drop(columns=ohe_source_columns, errors='ignore')
        df_movie = pd.concat([df_movie, ohe_df], axis=1)
    else:
        print("  Исходные колонки для OneHotEncoder не определены. Пропуск OHE.")


    # 8. Target Encoding
    target_encoder = artifacts.get('target_encoder')
    if not target_encoder: raise ValueError("target_encoder не найден в артефактах.")
    
    te_encoder_fit_cols = getattr(target_encoder, 'cols', []) 
    # Используем колонки, на которых TE был обучен И которые есть в cfg_categorical_cols (для консистентности)
    # И которые есть в текущем df_movie
    cols_for_te_transform = [col for col in te_encoder_fit_cols if col in df_movie.columns and col in cfg_categorical_cols]
    
    if target_encoder and cols_for_te_transform:
        for col_te in cols_for_te_transform:
            if df_movie[col_te].dtype != 'object' and df_movie[col_te].dtype != 'string':
                 df_movie[col_te] = df_movie[col_te].astype(str)
        
        transformed_te_vals = target_encoder.transform(df_movie[cols_for_te_transform])
        if isinstance(transformed_te_vals, pd.Series):
             df_movie[cols_for_te_transform[0]] = transformed_te_vals.values[0]
        elif isinstance(transformed_te_vals, pd.DataFrame):
             df_movie[cols_for_te_transform] = transformed_te_vals.values
    
    # 9. Reindex
    # ИСПОЛЬЗУЕМ КЛЮЧ "features" из вашего column_info.json
    final_model_features = column_info.get("features") 
    if not final_model_features: 
        raise ValueError("Ключ 'features' (список финальных признаков) отсутствует или пуст в column_info.json.")
    if not isinstance(final_model_features, list) or not final_model_features:
        raise ValueError("'features' должен быть непустым списком в column_info.json.")

    df_processed_for_model = pd.DataFrame(0, index=[0], columns=final_model_features)
    cols_to_transfer = [col for col in df_movie.columns if col in df_processed_for_model.columns]
    if cols_to_transfer:
        df_processed_for_model[cols_to_transfer] = df_movie[cols_to_transfer].values

    if df_processed_for_model.isnull().any().any():
        df_processed_for_model.fillna(0, inplace=True)
    
    # 10. Scaling
    scaler = artifacts.get('scaler')
    if not scaler: raise ValueError("scaler не найден в артефактах.")
    X_scaled = scaler.transform(df_processed_for_model)
    return X_scaled


def make_prediction(movie_data_dict):
    try:
        artifacts = get_artifacts() 
        if artifacts is None:
            raise RuntimeError("Не удалось загрузить артефакты модели.")
            
        processed_data = preprocess_input_data(movie_data_dict, artifacts)
        
        model = artifacts.get('model')
        if not model: raise ValueError("Модель не найдена в артефактах.")
        
        # ИСПОЛЬЗУЕМ КЛЮЧ "features" для имен признаков DMatrix
        final_model_feature_names = artifacts.get('column_info', {}).get("features")
        if not final_model_feature_names:
            raise ValueError("Список 'features' не найден в column_info для создания DMatrix.")

        if processed_data.shape[1] != len(final_model_feature_names):
            raise ValueError(f"Несоответствие количества признаков: ожидалось {len(final_model_feature_names)}, получено {processed_data.shape[1]}. Проверьте OHE и Reindex.")

        dmatrix_processed = xgb.DMatrix(processed_data, feature_names=final_model_feature_names)
        prediction_value = model.predict(dmatrix_processed)
        
        predicted_gross = round(float(prediction_value[0]))
        return predicted_gross
    except Exception as e:
        print(f"Ошибка в процессе предсказания: {e}")
        import traceback
        traceback.print_exc()
        raise