# model_service/predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import xgboost as xgb 
import logging

# --- Настройка логирования ---
LOG_LEVEL_FROM_ENV = os.environ.get("LOG_LEVEL", "DEBUG").upper()
numeric_level = getattr(logging, LOG_LEVEL_FROM_ENV, logging.DEBUG)
logging.basicConfig(level=numeric_level, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

MODEL_DIR = "saved_model" 
_loaded_artifacts_cache = None

# --- Вспомогательная функция для логирования DataFrame ---
def log_dataframe_details(df, name="DataFrame", max_rows=5, max_cols_to_show=20, log_level=logging.DEBUG): # Увеличил max_cols_to_show
    if not logger.isEnabledFor(log_level): return
    if not isinstance(df, pd.DataFrame): logger.log(log_level, f"'{name}' не DataFrame: {type(df)}"); return
    
    logger.log(log_level, f"--- DataFrame Details: '{name}' ---")
    logger.log(log_level, f"Форма: {df.shape}")
    logger.log(log_level, f"Колонки: {df.columns.tolist()}") # Логируем все имена колонок
    if df.empty: logger.log(log_level, "DataFrame пуст."); return

    if df.shape[0] > max_rows * 2 and max_rows > 0 :
        logger.log(log_level, f"Первые {max_rows} строк (до {max_cols_to_show} колонок):\n{df.head(max_rows).to_string(max_cols=max_cols_to_show)}")
        # logger.log(log_level, f"Последние {max_rows} строк (до {max_cols_to_show} колонок):\n{df.tail(max_rows).to_string(max_cols=max_cols_to_show)}")
    elif max_rows > 0:
        logger.log(log_level, f"Все строки ({df.shape[0]}) (до {max_cols_to_show} колонок):\n{df.to_string(max_cols=max_cols_to_show)}")

    nan_info = df.isnull().sum()
    nan_info = nan_info[nan_info > 0]
    if not nan_info.empty: logger.log(log_level, f"Колонки с NaN (количество):\n{nan_info.to_string()}")
    else: logger.log(log_level, "NaN в DataFrame отсутствуют.")
    logger.log(log_level, f"--- End DataFrame Details: '{name}' ---")


def convert_runtime_to_minutes(value):
    if pd.isna(value) or value == "": return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    logger.warning(f"Не удалось распознать run_time: '{value}'")
    return None

def get_artifacts(): # Логика загрузки остается такой же, как в test_pipeline.py
    global _loaded_artifacts_cache
    if _loaded_artifacts_cache is not None:
        logger.debug("Использование кэшированных артефактов.")
        return _loaded_artifacts_cache

    logger.info(f"Начало загрузки артефактов из директории: {MODEL_DIR}")
    artifacts = {}
    files_map = {
        "model_booster_json": "movie_box_office_model.json",
        "model_regressor_joblib": "movie_box_office_model.joblib",
        "numerical_imputer": "numerical_imputer.joblib",
        "categorical_imputer": "categorical_imputer.joblib",
        "onehot_encoder": "onehot_encoder.joblib",
        "target_encoder": "target_encoder.joblib",
        "scaler": "scaler.joblib",
        "column_info": "column_info.json"
    }
    try:
        for key, f_name in files_map.items():
            path = os.path.join(MODEL_DIR, f_name)
            if not os.path.exists(path):
                if key in ["model_booster_json", "model_regressor_joblib"]:
                    logger.warning(f"Файл модели {f_name} не найден.")
                    artifacts[key] = None; continue
                logger.error(f"Файл не найден: {path}"); raise FileNotFoundError(f"Не найден: {path}")
            
            if key == "model_booster_json": 
                model_obj = xgb.Booster(); model_obj.load_model(path); artifacts[key] = model_obj
                logger.info(f"Модель XGBoost Booster '{f_name}' загружена.")
            elif key == "model_regressor_joblib":
                artifacts[key] = joblib.load(path)
                logger.info(f"Модель XGBRegressor '{f_name}' загружена.")
            elif f_name.endswith(".joblib"):
                artifacts[key] = joblib.load(path)
                logger.info(f"Артефакт '{key}' ({f_name}) загружен.")
            elif key == "column_info":
                with open(path, "r", encoding='utf-8') as f: artifacts[key] = json.load(f)
                logger.info(f"Артефакт '{key}' ({f_name}) загружен.")
                logger.debug(f"column_info.json['features'][0:5]: {artifacts[key].get('features',[])[:5]}")
        
        if artifacts.get("model_booster_json"): artifacts["model"] = artifacts["model_booster_json"]
        elif artifacts.get("model_regressor_joblib"): artifacts["model"] = artifacts["model_regressor_joblib"]
        else: logger.error("Модель не загружена!"); raise ValueError("Модель не загружена.")
        
        artifacts.pop("model_booster_json", None); artifacts.pop("model_regressor_joblib", None)
        for k_essential in ["model", "numerical_imputer", "categorical_imputer", "onehot_encoder", "target_encoder", "scaler", "column_info"]:
            if k_essential not in artifacts or artifacts[k_essential] is None:
                logger.error(f"Критический артефакт '{k_essential}' не загружен!"); raise ValueError(f"'{k_essential}' не загружен.")
        logger.info("Все артефакты загружены и кэшированы.")
        _loaded_artifacts_cache = artifacts
        return artifacts
    except Exception as e:
        logger.critical(f"Ошибка загрузки артефактов: {e}", exc_info=True)
        _loaded_artifacts_cache = None; raise RuntimeError(f"Ошибка загрузки артефактов: {e}")

# Эта функция теперь будет максимально повторять preprocess_dataframe из test_pipeline.py
def preprocess_input_data(movie_data_dict: dict, artifacts: dict) -> tuple[np.ndarray, list]:
    logger.info("--- Начало предобработки входных данных (один фильм, логика из test_pipeline) ---")
    
    column_info = artifacts['column_info']
    df = pd.DataFrame([movie_data_dict]) # Создаем DataFrame из одной строки
    log_dataframe_details(df, "df_movie (начальный, из movie_data_dict)")

    # --- Шаг 0: Удаление начальных колонок (из column_info) ---
    # ВАЖНО: FastAPI сервис должен передавать словарь УЖЕ БЕЗ этих колонок.
    # Но для полной идентичности с test_pipeline.py, если они вдруг придут, удалим.
    initial_cols_to_drop = column_info.get("initial_columns_to_drop", [])
    logger.debug(f"Шаг 0: Initial_columns_to_drop (из config): {initial_cols_to_drop}")
    if initial_cols_to_drop:
        cols_present_to_drop = [col for col in initial_cols_to_drop if col in df.columns]
        if cols_present_to_drop:
            df = df.drop(columns=cols_present_to_drop)
            logger.info(f"  Удалены начальные колонки: {cols_present_to_drop}")
    log_dataframe_details(df, "df (после начального drop, если был)")


    # --- Шаг 1: Feature Engineering (опыт) ---
    logger.info("Шаг 1: Feature Engineering (опыт)")
    exp_maps = column_info.get("experience_maps", {})
    if not exp_maps: logger.warning("Словарь 'experience_maps' пуст или отсутствует в column_info!")
        
    personnel_cols_for_exp = column_info.get("personnel_cols_for_experience", [])
    logger.debug(f"  Personnel_cols_for_experience: {personnel_cols_for_exp}")
    for col in personnel_cols_for_exp:
        map_key = f"{col}_experience_map"
        current_personnel_map = exp_maps.get(map_key, {})
        # Для одной строки df.loc[0, col]
        name_in_movie = str(df.loc[0, col]) if col in df.columns and pd.notna(df.loc[0, col]) else ""
        exp_val = current_personnel_map.get(name_in_movie, 0)
        df[f"{col}_experience"] = exp_val # Присваиваем скалярное значение всей колонке (из одной строки)
        logger.debug(f"    {col}_experience для '{name_in_movie}': {exp_val}")

    actor_exp_cols_gen = []
    actor_prefix = column_info.get("actor_cols_for_experience_prefix", "main_actor_")
    for i in range(1, 5):
        col_actor = f"{actor_prefix}{i}"
        map_key_actor = f"{col_actor}_experience_map"
        current_actor_map = exp_maps.get(map_key_actor, {})
        name_in_movie = str(df.loc[0, col_actor]) if col_actor in df.columns and pd.notna(df.loc[0, col_actor]) else ""
        exp_val = current_actor_map.get(name_in_movie, 0)
        df[f"{col_actor}_experience"] = exp_val
        actor_exp_cols_gen.append(f"{col_actor}_experience")
        logger.debug(f"    {col_actor}_experience для '{name_in_movie}': {exp_val}")
    
    if actor_exp_cols_gen: 
        # Для одной строки sum(axis=1) вернет Series, берем .iloc[0]
        df["cast_popularity"] = df[actor_exp_cols_gen].sum(axis=1).iloc[0] 
    else: 
        df["cast_popularity"] = 0
    logger.debug(f"  cast_popularity: {df['cast_popularity'].iloc[0]}")
    log_dataframe_details(df, "df (после опыта)")


    # --- Шаг 2: Преобразование run_time ---
    logger.info("Шаг 2: Преобразование run_time")
    if 'run_time' in df.columns:
        # apply для DataFrame из одной строки вернет Series, присваиваем обратно
        df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes) 
        logger.debug(f"  run_time преобразован в: {df.loc[0, 'run_time']}")
    log_dataframe_details(df, "df (после run_time)")


    # --- Шаг 3: Numerical Imputation ---
    logger.info("Шаг 3: Numerical Imputation")
    num_imputer = artifacts['numerical_imputer']
    # Используем ТОЛЬКО тот список, что в column_info.json (предполагается, что он БЕЗ domestic/international/worldwide)
    num_features_to_impute = column_info.get("numerical_imputer_features_in", [])
    logger.info(f"  Ожидаемые колонки для Numerical Imputer (из column_info): {num_features_to_impute}")
    if not num_features_to_impute:
        logger.warning("Список numerical_imputer_features_in пуст в column_info! Numerical Imputation может быть некорректным или пропущен.")

    # Создаем недостающие числовые колонки, которые ожидает импьютер (из списка num_features_to_impute)
    for col in num_features_to_impute:
        if col not in df.columns: 
            df[col] = np.nan
            logger.debug(f"    Добавлена (NaN) '{col}' для numerical_imputer.")
            
    if num_features_to_impute: 
        df_subset_num = df[num_features_to_impute].copy() # Порядок важен
        log_dataframe_details(df_subset_num, "Данные для numerical_imputer (до transform)")
        imputed_num_vals = num_imputer.transform(df_subset_num)
        imputed_df_num = pd.DataFrame(imputed_num_vals, columns=num_features_to_impute, index=df.index)
        df[num_features_to_impute] = imputed_df_num
    else:
        logger.warning("  Список колонок для числовой импутации (num_features_to_impute) пуст. Пропуск.")
    log_dataframe_details(df, "df (после Numerical Imputation)")
    
    
    # --- Шаг 4: Grouped Imputation ---
    # Логика как в test_pipeline.py, но для одной строки
    logger.info("Шаг 4: Grouped Imputation")
    group_maps = column_info.get("grouped_imputation_maps", {}) 
    group_impute_cols = column_info.get("cols_for_grouped_imputation_source_director", []) 
    logger.debug(f"  Колонки для групповой импутации по 'director': {group_impute_cols}")
    if 'director' in df.columns and group_impute_cols: # Проверяем наличие director
        director_name_str = str(df.loc[0, 'director']) if pd.notna(df.loc[0, 'director']) else ""
        if director_name_str: # Только если есть имя режиссера
            for col_to_impute in group_impute_cols:
                if col_to_impute not in df.columns: df[col_to_impute] = np.nan
                
                if pd.isna(df.loc[0, col_to_impute]): # Заполняем только если пропуск
                    map_key = f"{col_to_impute}_director_mode_map"
                    current_col_map = group_maps.get(map_key, {})
                    mode_val = current_col_map.get(director_name_str) # Ключи в картах - строки
                    logger.debug(f"    Для '{col_to_impute}', карта '{map_key}', режиссер '{director_name_str}'. Найдена мода: {mode_val}")
                    if mode_val is not None: df.loc[0, col_to_impute] = mode_val
    log_dataframe_details(df, "df (после Grouped Imputation)")


    # --- Шаг 5: Fill 'Unknown' для специфических колонок ---
    logger.info("Шаг 5: Заполнение 'Unknown'")
    fill_unknown_cols = column_info.get("cols_to_fill_unknown_specific", []) 
    logger.debug(f"  Колонки для 'Unknown': {fill_unknown_cols}")
    if fill_unknown_cols:
        for col in fill_unknown_cols:
            if col not in df.columns: df[col] = np.nan
            # Применяем к одной строке
            current_val = df.loc[0, col] if col in df.columns else np.nan
            if pd.isna(current_val) or (isinstance(current_val, str) and current_val.strip() == ''):
                df.loc[0, col] = 'Unknown'
    log_dataframe_details(df, "df (после Fill Unknown)")


    # --- Шаг 6: Categorical Imputation (general) ---
    logger.info("Шаг 6: Categorical Imputation")
    cat_imputer = artifacts['categorical_imputer']
    cat_features_expected = column_info.get("categorical_imputer_features_in", []) # Из JSON
    if not cat_features_expected: # Запасной
        cat_features_expected = getattr(cat_imputer, 'feature_names_in_', [])
        if isinstance(cat_features_expected, np.ndarray): cat_features_expected = cat_features_expected.tolist()
    logger.debug(f"  Categorical Imputer ожидает: {cat_features_expected}")

    for col in cat_features_expected:
        if col not in df.columns: df[col] = np.nan
            
    if cat_features_expected:
        df_subset_cat = df[cat_features_expected].copy()
        log_dataframe_details(df_subset_cat, "Данные для categorical_imputer (до transform)")
        imputed_cat_vals = cat_imputer.transform(df_subset_cat)
        imputed_df_cat = pd.DataFrame(imputed_cat_vals, columns=cat_features_expected, index=df.index)
        df[cat_features_expected] = imputed_df_cat
    log_dataframe_details(df, "df (после Categorical Imputation)")


    # --- Шаг 7: One-Hot Encoding ---
    logger.info("Шаг 7: One-Hot Encoding")
    ohe = artifacts['onehot_encoder']
    ohe_input_columns = column_info.get("ohe_input_columns", []) # Из JSON
    if not ohe_input_columns: # Запасной
        if hasattr(ohe, 'feature_names_in_'): ohe_input_columns = ohe.feature_names_in_.tolist()
        else: logger.error("ohe_input_columns не найдены!"); raise ValueError("ohe_input_columns не найдены.")
    logger.debug(f"  Исходные колонки для OHE: {ohe_input_columns}")

    for col in ohe_input_columns:
        if col not in df.columns: df[col] = "Unknown" 
        current_val_ohe = df.loc[0, col] if col in df.columns else "Unknown"
        if pd.isna(current_val_ohe) or (isinstance(current_val_ohe, str) and current_val_ohe.strip() == ""):
            df.loc[0, col] = "Unknown"
        if col in df.columns and df[col].dtype != 'object' and df[col].dtype != 'string':
             df[col] = df[col].astype(str)
    
    if ohe_input_columns: 
        log_dataframe_details(df[ohe_input_columns], f"Данные для OHE (колонки {ohe_input_columns})")
        ohe_transformed_data = ohe.transform(df[ohe_input_columns])
        ohe_feature_names = ohe.get_feature_names_out(ohe_input_columns)
        if not ohe_feature_names.size > 0 : logger.error("Имена OHE пусты!"); raise RuntimeError("Имена OHE пусты.")
        logger.debug(f"  OHE сгенерировал {len(ohe_feature_names)} колонок.")
        ohe_df = pd.DataFrame(ohe_transformed_data.toarray() if hasattr(ohe_transformed_data, "toarray") else ohe_transformed_data, 
                              columns=ohe_feature_names, index=df.index)
        df = df.drop(columns=ohe_input_columns, errors='ignore')
        df = pd.concat([df, ohe_df], axis=1)
    log_dataframe_details(df, "df (после OHE)")


    # --- Шаг 8: Target Encoding ---
    logger.info("Шаг 8: Target Encoding")
    target_encoder = artifacts['target_encoder']
    te_input_columns = column_info.get("target_encoding_columns", []) # Из JSON
    if not te_input_columns: # Запасной
        if hasattr(target_encoder, 'cols'): te_input_columns = target_encoder.cols
        else: logger.error("target_encoding_columns не найдены!"); raise ValueError("target_encoding_columns не найдены.")
    logger.debug(f"  Исходные колонки для Target Encoding: {te_input_columns}")

    cols_for_te_transform = []
    for col in te_input_columns:
        if col not in df.columns: df[col] = "Unknown"
        current_val_te = df.loc[0, col] if col in df.columns else "Unknown"
        if pd.isna(current_val_te) or (isinstance(current_val_te, str) and current_val_te.strip() == ""):
            df.loc[0, col] = "Unknown" 
        if col in df.columns and df[col].dtype != 'object' and df[col].dtype != 'string':
             df[col] = df[col].astype(str)
        cols_for_te_transform.append(col) # Добавляем, даже если создали как 'Unknown'
            
    if target_encoder and cols_for_te_transform: # Убедимся, что есть что трансформировать
        log_dataframe_details(df[cols_for_te_transform], f"Данные для Target Encoder (колонки {cols_for_te_transform})")
        transformed_te_vals = target_encoder.transform(df[cols_for_te_transform])
        df[cols_for_te_transform] = transformed_te_vals # category_encoders возвращает DataFrame
    log_dataframe_details(df, "df (после Target Encoding)")
    
    # --- Шаг 9: Reindex ---
    logger.info("Шаг 9: Reindex")
    final_model_features = column_info.get("features") 
    if not isinstance(final_model_features, list) or not final_model_features:
        logger.error("'features' из column_info некорректен!"); raise ValueError("'features' должен быть непустым списком.")
    logger.debug(f"  Ожидаемые финальные признаки (всего {len(final_model_features)}).")
    
    df_processed_for_model = pd.DataFrame(0.0, index=[0], columns=final_model_features) # Используем 0.0 для float dtype
    common_cols = df_processed_for_model.columns.intersection(df.columns)
    if not common_cols.empty:
        # Присваиваем значения. df.loc[0, common_cols] - это Series.
        # df_processed_for_model.loc[0, common_cols] - это тоже Series.
        df_processed_for_model.loc[0, common_cols] = df.loc[0, common_cols].values 
    
    if df_processed_for_model.isnull().any().any():
        nan_cols_final = df_processed_for_model.columns[df_processed_for_model.iloc[0].isnull()].tolist()
        logger.warning(f"  NaN обнаружены в df_processed_for_model перед масштабированием: {nan_cols_final}. Заполнение нулями.")
        df_processed_for_model.fillna(0, inplace=True)
    log_dataframe_details(df_processed_for_model, "df_processed_for_model (после Reindex, перед Scaling)")
    
    # --- Шаг 10: Scaling ---
    logger.info("Шаг 10: Scaling")
    scaler = artifacts['scaler']
    X_scaled_np = scaler.transform(df_processed_for_model)
    logger.debug(f"  Данные после масштабирования (форма): {X_scaled_np.shape}")
    logger.info("--- Предобработка входных данных (один фильм) ЗАВЕРШЕНА ---")
    return X_scaled_np, final_model_features


def make_prediction(movie_data_dict: dict) -> float:
    logger.info(f"Получен запрос на предсказание (ключевые поля из movie_data_dict): "
                f"{ {k:v for k,v in movie_data_dict.items() if k in ['movie_title', 'director', 'budget', 'movie_year']} }")
    try:
        artifacts = get_artifacts() 
        if artifacts is None: logger.error("Артефакты не загружены!"); raise RuntimeError("Артефакты не загружены.")
            
        processed_data_np, feature_names_for_dmatrix = preprocess_input_data(movie_data_dict, artifacts)
        
        log_dataframe_details(pd.DataFrame(processed_data_np, columns=feature_names_for_dmatrix), 
                              "processed_data_np (готово для DMatrix)", log_level=logging.INFO) # Выводим это на INFO
        
        model = artifacts.get('model')
        if not model: logger.error("Модель не найдена!"); raise ValueError("Модель не найдена.")
        
        # feature_names_for_dmatrix уже содержит правильный список из preprocess_input_data (это final_model_features)
        if processed_data_np.shape[1] != len(feature_names_for_dmatrix):
            msg = (f"Несоответствие признаков: DMatrix ожидает {len(feature_names_for_dmatrix)} ({feature_names_for_dmatrix[:3]}...), "
                   f"получено {processed_data_np.shape[1]}.")
            logger.error(msg); raise ValueError(msg)

        logger.info("Создание DMatrix и выполнение предсказания моделью XGBoost...")
        dmatrix_processed = xgb.DMatrix(processed_data_np, feature_names=feature_names_for_dmatrix)
        prediction_value = model.predict(dmatrix_processed)
        logger.info(f"Предсказание моделью (сырое): {prediction_value}")
        
        predicted_gross = round(float(prediction_value[0]))
        logger.info(f"Финальное предсказание (округленное): ${predicted_gross:,.0f}")
        return predicted_gross
    except Exception as e:
        logger.error(f"Ошибка в процессе предсказания: {e}", exc_info=True)
        raise