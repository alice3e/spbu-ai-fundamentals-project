# predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import re
import os

MODEL_DIR = "saved_model" # Директория с артефактами

# Глобальная переменная для хранения загруженных артефактов
# Это позволит загружать их только один раз
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
    """
    Загружает артефакты модели, если они еще не загружены, и кэширует их.
    Возвращает словарь с артефактами.
    """
    global _loaded_artifacts_cache
    if _loaded_artifacts_cache is not None:
        # print("Использование кэшированных артефактов.")
        return _loaded_artifacts_cache

    print(f"Загрузка артефактов из директории: {MODEL_DIR}")
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
            path = os.path.join(MODEL_DIR, f_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден файл артефакта: {path}")
            
            if f_name.endswith(".joblib"):
                artifacts[key] = joblib.load(path)
            elif f_name.endswith(".json"):
                with open(path, "r", encoding='utf-8') as f:
                    artifacts[key] = json.load(f)
        
        print("Артефакты успешно загружены и кэшированы.")
        _loaded_artifacts_cache = artifacts
        
        # Опционально: вывести информацию о загруженных компонентах
        # if hasattr(artifacts['numerical_imputer'], 'feature_names_in_'):
        #     print(f"  Numerical Imputer обучен на: {artifacts['numerical_imputer'].feature_names_in_}")
        # if hasattr(artifacts['categorical_imputer'], 'feature_names_in_'):
        #     print(f"  Categorical Imputer обучен на: {artifacts['categorical_imputer'].feature_names_in_}")

        return artifacts
    except Exception as e:
        print(f"Критическая ошибка при загрузке артефактов: {e}")
        _loaded_artifacts_cache = None # Сбрасываем кэш при ошибке
        raise RuntimeError(f"Не удалось загрузить артефакты: {e}")


def preprocess_input_data(movie_data_dict, artifacts):
    """
    Предобрабатывает входные данные одного фильма для модели.
    movie_data_dict: словарь с данными фильма.
    artifacts: словарь с загруженными артефактами.
    """
    # print("Preprocessing input data...")
    column_info = artifacts['column_info']
    df_movie = pd.DataFrame([movie_data_dict])

    # 1. Feature Engineering (опыт)
    exp_maps = column_info.get("experience_maps", {})
    personnel_cols = ["director", "writer", "producer", "composer", "cinematographer"]
    for col in personnel_cols:
        map_key = f"{col}_experience_map"
        name_in_movie = str(df_movie.loc[0, col]) if col in df_movie.columns and pd.notna(df_movie.loc[0, col]) else None
        exp_val = exp_maps.get(map_key, {}).get(name_in_movie, 0) if name_in_movie else 0
        df_movie[f"{col}_experience"] = exp_val

    actor_exp_cols_gen = []
    for i in range(1, 5):
        col = f"main_actor_{i}"
        map_key = f"{col}_experience_map"
        name_in_movie = str(df_movie.loc[0, col]) if col in df_movie.columns and pd.notna(df_movie.loc[0, col]) else None
        exp_val = exp_maps.get(map_key, {}).get(name_in_movie, 0) if name_in_movie else 0
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
    
    for col in num_features_expected:
        if col not in df_movie.columns: df_movie[col] = np.nan # Создаем, если нет
    
    if num_features_expected: 
        df_movie_subset_num = df_movie[num_features_expected] # Выбираем и упорядочиваем колонки
        imputed_num_vals = num_imputer.transform(df_movie_subset_num)
        for i_num, col_name_num in enumerate(num_features_expected):
            df_movie[col_name_num] = imputed_num_vals[0, i_num]
    
    # 4. Grouped Imputation
    group_maps = column_info.get("grouped_imputation_maps", {})
    group_impute_cols = column_info.get("group_impute_director_dependent_cols", [])
    director_name_str = str(df_movie.loc[0, 'director']) if 'director' in df_movie.columns and pd.notna(df_movie.loc[0, 'director']) else None
    if director_name_str:
        for col in group_impute_cols:
            if col not in df_movie.columns: df_movie[col] = np.nan
            if pd.isna(df_movie.loc[0, col]):
                map_key = f"{col}_director_mode_map"
                mode_val = group_maps.get(map_key, {}).get(director_name_str) # Ключи в картах - строки
                if mode_val is not None: df_movie.loc[0, col] = mode_val
    
    # 5. Fill 'Unknown' for specific columns
    fill_unknown_cols = column_info.get("fill_unknown_cols", [])
    for col in fill_unknown_cols:
        if col not in df_movie.columns: df_movie[col] = np.nan
        # Заполняем 'Unknown', если значение NaN или пустая строка
        if pd.isna(df_movie.loc[0, col]) or (isinstance(df_movie.loc[0, col], str) and df_movie.loc[0, col].strip() == ''):
            df_movie.loc[0, col] = 'Unknown'

    # 6. Categorical Imputation (general)
    cat_imputer = artifacts['categorical_imputer']
    cat_features_expected = getattr(cat_imputer, 'feature_names_in_', [])
    if isinstance(cat_features_expected, np.ndarray): cat_features_expected = cat_features_expected.tolist()

    for col in cat_features_expected:
        if col not in df_movie.columns: df_movie[col] = np.nan
    
    if cat_features_expected:
        df_movie_subset_cat = df_movie[cat_features_expected] # Выбираем и упорядочиваем
        imputed_cat_vals = cat_imputer.transform(df_movie_subset_cat)
        for i_cat, col_name_cat in enumerate(cat_features_expected):
            df_movie[col_name_cat] = imputed_cat_vals[0, i_cat]

    # 7. One-Hot Encoding
    ohe_cols = column_info.get("one_hot_encoded_cols", [])
    actual_ohe_cols = [col for col in ohe_cols if col in df_movie.columns] # Колонки, реально присутствующие в df_movie
    if actual_ohe_cols:
        df_movie = pd.get_dummies(df_movie, columns=actual_ohe_cols, dummy_na=False)

    # 8. Target Encoding
    target_encoder = artifacts['target_encoder']
    te_encoder_fit_cols = getattr(target_encoder, 'cols', []) # Колонки, на которых TE был обучен
    cols_for_te_transform = [col for col in te_encoder_fit_cols if col in df_movie.columns] # Из них те, что есть в df_movie
    if target_encoder and cols_for_te_transform:
        transformed_te_vals = target_encoder.transform(df_movie[cols_for_te_transform])
        if isinstance(transformed_te_vals, pd.Series): # Если одна колонка
             df_movie[cols_for_te_transform[0]] = transformed_te_vals.values[0]
        elif isinstance(transformed_te_vals, pd.DataFrame): # Если несколько колонок
             df_movie[cols_for_te_transform] = transformed_te_vals.values
    
    # 9. Reindex to match model's expected features
    final_features = column_info.get("final_model_features", [])
    if not final_features: 
        raise ValueError("final_model_features не найдены в column_info.")
    
    # Создаем DataFrame с нужными колонками и заполняем нулями
    df_processed_for_model = pd.DataFrame(0, index=[0], columns=final_features)
    
    # Заполняем значениями из df_movie, где колонки совпадают
    # Колонки, которые есть в df_movie И в final_features
    cols_to_transfer = [col for col in df_movie.columns if col in df_processed_for_model.columns]
    if cols_to_transfer:
        df_processed_for_model[cols_to_transfer] = df_movie[cols_to_transfer].values

    # Финальная проверка на NaN и заполнение, если что-то просочилось
    if df_processed_for_model.isnull().any().any():
        # print("  Предупреждение: Неожиданные NaN перед масштабированием. Заполнение нулями.")
        df_processed_for_model.fillna(0, inplace=True)
    
    # 10. Scaling
    X_scaled = artifacts['scaler'].transform(df_processed_for_model)
    return X_scaled


def make_prediction(movie_data_dict):
    """
    Выполняет полное предсказание для одного фильма.
    Загружает артефакты (если не загружены), предобрабатывает данные и делает предсказание.
    """
    try:
        artifacts = get_artifacts() # Получаем кэшированные или загружаем новые
        if artifacts is None:
            raise RuntimeError("Не удалось загрузить артефакты модели.")
            
        processed_data = preprocess_input_data(movie_data_dict, artifacts)
        model = artifacts['model']
        prediction_value = model.predict(processed_data)
        
        # Округляем до целого, так как это доллары
        predicted_gross = round(float(prediction_value[0]))
        return predicted_gross
    except Exception as e:
        print(f"Ошибка в процессе предсказания: {e}")
        import traceback
        traceback.print_exc()
        raise # Перевыбрасываем исключение, чтобы Flask мог его поймать