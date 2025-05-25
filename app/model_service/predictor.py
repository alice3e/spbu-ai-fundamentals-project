# model_service/predictor.py
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import xgboost as xgb 
import logging
from typing import Dict, Any, Tuple, List, Optional # Добавил Optional

# --- Настройка логирования ---
LOG_LEVEL_FROM_ENV = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, LOG_LEVEL_FROM_ENV, logging.INFO)

logging.basicConfig(level=numeric_level, 
                    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model") 
_loaded_artifacts_cache: Dict[str, Any] = {}

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame", max_rows: int = 3, max_cols: int = 10, log_level: int = logging.DEBUG):
    if not logger.isEnabledFor(log_level): return
    if not isinstance(df, pd.DataFrame): 
        logger.log(log_level, f"'{name}' не DataFrame: {type(df)}")
        return
    
    logger.log(log_level, f"Инфо о '{name}': Форма: {df.shape}")
    if df.empty: 
        logger.log(log_level, f"  '{name}' пуст.")
        return

    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty: 
        logger.log(log_level, f"  Колонки с NaN (в '{name}'):\n{nan_cols.to_string()}")
    
    if df.shape[1] <= max_cols :
        logger.log(log_level, f"  Содержимое '{name}' (все {df.shape[1]} колонок):\n{df.head(max_rows).to_string(max_cols=max_cols)}")
    else:
        num_cols_to_show_each_side = max_cols // 2
        start_cols_indices = list(range(min(num_cols_to_show_each_side, df.shape[1])))
        end_cols_count = min(num_cols_to_show_each_side, df.shape[1] - len(start_cols_indices))
        end_cols_indices = list(range(df.shape[1] - end_cols_count, df.shape[1])) if end_cols_count > 0 else []
        selected_col_indices = sorted(list(set(start_cols_indices + end_cols_indices)))

        if selected_col_indices:
            logger.log(log_level, f"  Первые {min(max_rows, df.shape[0])} строк (выборочно {len(selected_col_indices)} из {df.shape[1]} колонок) из '{name}':\n{df.iloc[:min(max_rows, df.shape[0]), selected_col_indices].to_string()}")
        elif df.shape[1] == 0:
             logger.log(log_level, f"  '{name}' не имеет колонок для отображения.")
        else: # df.shape[1] > 0 но selected_col_indices пуст (маловероятно)
             logger.log(log_level, f"  Не удалось выбрать колонки для отображения из '{name}'.")


def convert_runtime_to_minutes(value: Any) -> Optional[int]:
    if pd.isna(value) or value == "": 
        return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    logger.warning(f"Не удалось распознать run_time: '{value}', возвращаем None.")
    return None

def get_artifacts() -> Dict[str, Any]:
    global _loaded_artifacts_cache
    if _loaded_artifacts_cache:
        logger.debug("Использование кэшированных артефактов.")
        return _loaded_artifacts_cache

    logger.info(f"Начало загрузки артефактов из директории: {MODEL_DIR}")
    if not os.path.isdir(MODEL_DIR):
        logger.critical(f"Директория с артефактами '{MODEL_DIR}' не найдена!")
        raise RuntimeError(f"Директория с артефактами '{MODEL_DIR}' не найдена.")
        
    artifacts: Dict[str, Any] = {}
    # Определяем имена файлов артефактов
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
    
    # Сначала загружаем column_info, так как он определяет необходимость других артефактов
    column_info_path = os.path.join(MODEL_DIR, files_map["column_info"])
    if not os.path.exists(column_info_path):
        logger.critical(f"Критический файл column_info.json не найден по пути: {column_info_path}")
        raise RuntimeError(f"column_info.json не найден: {column_info_path}")
    try:
        with open(column_info_path, "r", encoding='utf-8') as f: 
            artifacts["column_info"] = json.load(f)
        logger.info(f"Артефакт 'column_info' ({files_map['column_info']}) успешно загружен.")
    except Exception as e_load_ci:
        logger.critical(f"Ошибка при загрузке column_info.json: {e_load_ci}", exc_info=True)
        raise RuntimeError(f"Ошибка загрузки column_info.json: {e_load_ci}")

    column_info_content = artifacts["column_info"]

    # Определяем, какие трансформеры действительно нужны на основе column_info
    required_transformers = {}
    if column_info_content.get("numerical_imputer_features_in"):
        required_transformers["numerical_imputer"] = files_map["numerical_imputer"]
    if column_info_content.get("categorical_imputer_features_in"):
        required_transformers["categorical_imputer"] = files_map["categorical_imputer"]
    if column_info_content.get("ohe_input_columns"):
        required_transformers["onehot_encoder"] = files_map["onehot_encoder"]
    if column_info_content.get("target_encoding_columns"):
        required_transformers["target_encoder"] = files_map["target_encoder"]
    # scaler всегда нужен
    required_transformers["scaler"] = files_map["scaler"]


    # Загружаем модели и необходимые трансформеры
    items_to_load = {
        "model_booster_json": files_map["model_booster_json"],
        "model_regressor_joblib": files_map["model_regressor_joblib"],
        **required_transformers # Добавляем только нужные трансформеры
    }

    for key, f_name in items_to_load.items():
        path = os.path.join(MODEL_DIR, f_name)
        logger.debug(f"Попытка загрузки: {key} из {path}")
        if not os.path.exists(path):
            if key in ["model_booster_json", "model_regressor_joblib"]:
                logger.warning(f"Файл модели '{f_name}' не найден.")
                artifacts[key] = None # Пометим как None, выбор модели будет ниже
            elif key in required_transformers: # Если это требуемый трансформер
                logger.critical(f"Требуемый артефакт '{key}' ({f_name}) не найден по пути: {path}")
                raise RuntimeError(f"Требуемый артефакт '{key}' не найден: {path}")
            # else: # Этот случай не должен возникнуть, т.к. грузим только из items_to_load
            #     logger.warning(f"Артефакт '{key}' ({f_name}) не найден, но не помечен как требуемый.")
            #     artifacts[key] = None
        else:
            try:
                if key == "model_booster_json": 
                    model_obj = xgb.Booster(); model_obj.load_model(path); artifacts[key] = model_obj
                    logger.info(f"Модель XGBoost Booster '{f_name}' успешно загружена.")
                elif key == "model_regressor_joblib":
                    artifacts[key] = joblib.load(path)
                    logger.info(f"Модель XGBRegressor '{f_name}' успешно загружена.")
                elif f_name.endswith(".joblib"): # Для трансформеров
                    artifacts[key] = joblib.load(path)
                    logger.info(f"Артефакт '{key}' ({f_name}) успешно загружен.")
                # column_info уже загружен
            except Exception as e_load:
                logger.error(f"Ошибка при загрузке артефакта {key} из {path}: {e_load}", exc_info=True)
                if key in ["model_booster_json", "model_regressor_joblib"] or key in required_transformers:
                    raise RuntimeError(f"Ошибка загрузки критического артефакта {key}: {e_load}")
                artifacts[key] = None

    # Выбор модели
    if artifacts.get("model_booster_json"):
        artifacts["model"] = artifacts["model_booster_json"]
        logger.info("Используется модель XGBoost Booster (из .json).")
    elif artifacts.get("model_regressor_joblib"):
        artifacts["model"] = artifacts["model_regressor_joblib"]
        logger.info("Используется модель XGBRegressor (из .joblib).")
    else:
        logger.critical("Ни одна из версий модели (JSON/Joblib) не была найдена или загружена.")
        raise ValueError("Не удалось загрузить основную модель.")
    
    artifacts.pop("model_booster_json", None) # Удаляем, если был None или уже скопирован в "model"
    artifacts.pop("model_regressor_joblib", None)

    # Финальная проверка, что все, что должно быть загружено (модель, column_info, scaler + нужные трансформеры), загружено
    final_check_list = ["model", "column_info", "scaler"] + list(required_transformers.keys())
    for k_check in list(set(final_check_list)): # set для уникальности
        if k_check not in artifacts or artifacts[k_check] is None:
            # Это должно было быть поймано ранее, но для подстраховки
            logger.critical(f"Финальная проверка: Критический артефакт '{k_check}' отсутствует в загруженных артефактах.")
            raise ValueError(f"Финальная проверка: Артефакт '{k_check}' не был успешно загружен.")

    _loaded_artifacts_cache = artifacts
    logger.info("Все необходимые артефакты (с учетом column_info) успешно загружены и кэшированы.")
    return artifacts

def preprocess_input_data(movie_data_dict: Dict[str, Any], artifacts: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    logger.info("--- НАЧАЛО ПРЕДОБРАБОТКИ ВХОДНЫХ ДАННЫХ (один фильм) ---")
    df = pd.DataFrame([movie_data_dict])
    log_dataframe_info(df, "df_movie_input (из словаря)")

    column_info = artifacts['column_info']

    # --- Шаг 0: Удаление начальных колонок ---
    initial_cols_to_drop = column_info.get("initial_columns_to_drop", [])
    if initial_cols_to_drop:
        cols_present_in_df = [col for col in initial_cols_to_drop if col in df.columns]
        if cols_present_in_df:
            df = df.drop(columns=cols_present_in_df)
            logger.debug(f"Удалены начальные колонки: {cols_present_in_df}")

    # --- Шаг 1: Feature Engineering (опыт) ---
    logger.debug("Шаг 1: Feature Engineering (опыт)")
    exp_maps = column_info.get("experience_maps", {})
    personnel_cols = column_info.get("personnel_cols_for_experience", [])
    for col in personnel_cols:
        map_key = f"{col}_experience_map"
        current_personnel_map = exp_maps.get(map_key, {})
        # Создаем колонку опыта, даже если исходной колонки нет, заполняем 0
        df[f"{col}_experience"] = 0 
        if col in df.columns and pd.notna(df.loc[0, col]):
            person_name = str(df.loc[0, col])
            df.loc[0, f"{col}_experience"] = current_personnel_map.get(person_name, 0)
        elif col not in df.columns:
            logger.debug(f"Колонка '{col}' для опыта персонала отсутствует, {col}_experience = 0.")
        # Если col есть, но значение NaN, {col}_experience останется 0

    actor_exp_cols_generated = []
    actor_prefix = column_info.get("actor_cols_for_experience_prefix", "main_actor_")
    for i in range(1, 5):
        col_actor = f"{actor_prefix}{i}"
        map_key_actor = f"{col_actor}_experience_map"
        current_actor_map = exp_maps.get(map_key_actor, {})
        exp_col_name = f"{col_actor}_experience"
        df[exp_col_name] = 0 # Инициализируем нулем
        if col_actor in df.columns and pd.notna(df.loc[0, col_actor]):
            actor_name = str(df.loc[0, col_actor])
            df.loc[0, exp_col_name] = current_actor_map.get(actor_name, 0)
        actor_exp_cols_generated.append(exp_col_name) 
    
    # cast_popularity создается всегда, даже если все exp_cols нули или не были сгенерированы
    # (хотя они всегда генерируются с i=1..4)
    df["cast_popularity"] = df[actor_exp_cols_generated].sum(axis=1).iloc[0] if actor_exp_cols_generated else 0
    log_dataframe_info(df, "df (после опыта)")

    # --- Шаг 2: Преобразование run_time ---
    logger.debug("Шаг 2: Преобразование run_time")
    if 'run_time' in df.columns:
        df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes)
    # Если 'run_time' нет, колонка не будет создана на этом шаге.
    # Numerical Imputer должен будет с этим справиться, если 'run_time' в его списке.
    log_dataframe_info(df, "df (после run_time)")

    # --- Шаг 3: Numerical Imputation ---
    logger.debug("Шаг 3: Numerical Imputation")
    num_imputer = artifacts.get('numerical_imputer')
    num_features_expected = column_info.get("numerical_imputer_features_in", [])
    if num_imputer and num_features_expected:
        for col in num_features_expected: 
            if col not in df.columns: df[col] = np.nan # Создаем с NaN, если отсутствует
        
        df_subset_num = df[num_features_expected].copy()
        imputed_values = num_imputer.transform(df_subset_num)
        df[num_features_expected] = pd.DataFrame(imputed_values, columns=num_features_expected, index=df.index)
    elif num_features_expected and not num_imputer: # Если список есть, а импьютера нет (ошибка загрузки)
         logger.error(f"Numerical imputer не загружен, но ожидались фичи: {num_features_expected}. ОСТАНОВКА.")
         raise ValueError(f"Numerical imputer не загружен, но нужен для {num_features_expected}")
    log_dataframe_info(df, "df (после Numerical Imputation)")
    
    # --- Шаг 4: Grouped Imputation ---
    logger.debug("Шаг 4: Grouped Imputation")
    group_maps = column_info.get("grouped_imputation_maps", {}) 
    group_impute_cols = column_info.get("cols_for_grouped_imputation_source_director", []) 
    if 'director' in df.columns and group_impute_cols and pd.notna(df.loc[0, 'director']):
        director_name_str = str(df.loc[0, 'director'])
        for col_to_impute in group_impute_cols:
            if col_to_impute not in df.columns: df[col_to_impute] = np.nan
            
            if pd.isna(df.loc[0, col_to_impute]): 
                map_key = f"{col_to_impute}_director_mode_map"
                current_col_map = group_maps.get(map_key, {})
                mode_val = current_col_map.get(director_name_str) 
                if mode_val is not None: df.loc[0, col_to_impute] = mode_val
    log_dataframe_info(df, "df (после Grouped Imputation)")

    # --- Шаг 5: Fill 'Unknown' для специфических колонок ---
    logger.debug("Шаг 5: Fill 'Unknown' для специфических колонок")
    fill_unknown_cols = column_info.get("cols_to_fill_unknown_specific", []) 
    if fill_unknown_cols:
        for col in fill_unknown_cols:
            if col not in df.columns: df[col] = 'Unknown' # Если колонки нет, создаем сразу с 'Unknown'
            else: # Если колонка есть
                current_val = df.loc[0, col]
                if pd.isna(current_val) or (isinstance(current_val, str) and current_val.strip() == ''):
                    df.loc[0, col] = 'Unknown'
    log_dataframe_info(df, "df (после Fill Unknown)")

    # --- Шаг 6: Categorical Imputation (general) ---
    logger.debug("Шаг 6: Categorical Imputation (general)")
    cat_imputer = artifacts.get('categorical_imputer')
    cat_features_expected = column_info.get("categorical_imputer_features_in", [])
    if cat_imputer and cat_features_expected:
        for col in cat_features_expected: 
            if col not in df.columns: df[col] = np.nan # Создаем с NaN
        
        df_subset_cat = df[cat_features_expected].copy()
        imputed_cat_values = cat_imputer.transform(df_subset_cat)
        df[cat_features_expected] = pd.DataFrame(imputed_cat_values, columns=cat_features_expected, index=df.index)
    elif cat_features_expected and not cat_imputer:
        logger.error(f"Categorical imputer не загружен, но ожидались фичи: {cat_features_expected}. ОСТАНОВКА.")
        raise ValueError(f"Categorical imputer не загружен, но нужен для {cat_features_expected}")
    log_dataframe_info(df, "df (после Categorical Imputation)")

    # --- Шаг 7: One-Hot Encoding ---
    logger.debug("Шаг 7: One-Hot Encoding")
    ohe = artifacts.get('onehot_encoder')
    ohe_input_columns = column_info.get("ohe_input_columns", [])
    if ohe and ohe_input_columns:
        for col in ohe_input_columns: 
            if col not in df.columns: df[col] = "Unknown" # Создаем с "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str) # Заполняем NaN и приводим к строке
        
        ohe_transformed_data = ohe.transform(df[ohe_input_columns])
        ohe_feature_names = ohe.get_feature_names_out(ohe_input_columns)
        ohe_df = pd.DataFrame(ohe_transformed_data.toarray() if hasattr(ohe_transformed_data, "toarray") else ohe_transformed_data, 
                              columns=ohe_feature_names, index=df.index)
        df = df.drop(columns=ohe_input_columns, errors='ignore') # Удаляем исходные OHE колонки
        df = pd.concat([df, ohe_df], axis=1)
    elif ohe_input_columns and not ohe:
        logger.error(f"OneHotEncoder не загружен, но ожидались колонки: {ohe_input_columns}. ОСТАНОВКА.")
        raise ValueError(f"OneHotEncoder не загружен, но нужен для {ohe_input_columns}")
    log_dataframe_info(df, "df (после OHE)")

    # --- Шаг 8: Target Encoding ---
    logger.debug("Шаг 8: Target Encoding")
    target_encoder = artifacts.get('target_encoder')
    te_input_columns = column_info.get("target_encoding_columns", [])
    if target_encoder and te_input_columns:
        cols_for_te_transform = []
        for col in te_input_columns: 
            if col not in df.columns: df[col] = "Unknown" # Создаем с "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)
            cols_for_te_transform.append(col)
        
        if cols_for_te_transform: # Только если есть что трансформировать
            transformed_te_vals = target_encoder.transform(df[cols_for_te_transform])
            df[cols_for_te_transform] = transformed_te_vals # category_encoders возвращает DataFrame
    elif te_input_columns and not target_encoder:
        logger.error(f"TargetEncoder не загружен, но ожидались колонки: {te_input_columns}. ОСТАНОВКА.")
        raise ValueError(f"TargetEncoder не загружен, но нужен для {te_input_columns}")
    log_dataframe_info(df, "df (после Target Encoding)")
    
    # --- Шаг 9: Reindex ---
    logger.debug("Шаг 9: Reindex")
    final_model_features = column_info.get("features") 
    if not isinstance(final_model_features, list) or not final_model_features:
        logger.critical("'features' из column_info некорректен или пуст!"); 
        raise ValueError("'features' должен быть непустым списком.")
    
    # Создаем DataFrame с нужными колонками и порядком, dtype=float для scaler
    df_processed_for_model = pd.DataFrame(0.0, index=df.index, columns=final_model_features) 
    
    common_cols = df_processed_for_model.columns.intersection(df.columns)
    logger.debug(f"Общие колонки для переноса в финальный DataFrame (всего {len(common_cols)}): {list(common_cols)[:5]}...")
    if not common_cols.empty:
        # Присваиваем значения, сохраняя dtype из df_processed_for_model, где это возможно
        # Однако, если в df есть строки, а в df_processed_for_model - float, будет ошибка или FutureWarning
        # Важно, чтобы к этому моменту все колонки, которые должны быть числами, уже были числами.
        for col_name in common_cols:
            try:
                df_processed_for_model[col_name] = df[col_name].astype(df_processed_for_model[col_name].dtype)
            except ValueError as e_type:
                # Если прямое приведение типа не удалось (например, строка в float)
                logger.error(f"Ошибка приведения типа для колонки '{col_name}' при Reindex: {e_type}. "
                             f"Значение из df: {df.loc[df.index[0], col_name]}, тип {type(df.loc[df.index[0], col_name])}. "
                             f"Ожидался тип {df_processed_for_model[col_name].dtype}. "
                             "Это указывает на проблему в предыдущих шагах предобработки (кодирование).")
                raise ValueError(f"Ошибка типа для '{col_name}' при Reindex. Предыдущие шаги не преобразовали ее в число.") from e_type
            except Exception as e_reindex: # Другие возможные ошибки
                 logger.error(f"Непредвиденная ошибка при копировании колонки '{col_name}' при Reindex: {e_reindex}")
                 raise

    missing_in_df_but_expected = set(final_model_features) - set(df.columns)
    if missing_in_df_but_expected: 
        logger.warning(f"  {len(missing_in_df_but_expected)} final_features отсутствовали в обработанном df и будут 0.0 (после инициализации df_processed_for_model): {list(missing_in_df_but_expected)[:3]}...")
    
    if df_processed_for_model.isnull().any().any():
        nan_cols_final = df_processed_for_model.columns[df_processed_for_model.isnull().any()].tolist()
        logger.warning(f"  NaN обнаружены в df_processed_for_model перед масштабированием (после копирования и инициализации нулями): {nan_cols_final}. "
                       "Это не должно происходить, если все колонки были правильно инициализированы или заполнены. "
                       "Принудительно заполняем 0.0.")
        df_processed_for_model.fillna(0.0, inplace=True)
    log_dataframe_info(df_processed_for_model, "df_processed_for_model (после Reindex, перед Scaling)")
    
    # --- Шаг 10: Scaling ---
    logger.debug("Шаг 10: Scaling")
    scaler = artifacts.get('scaler') # scaler всегда должен быть
    if not scaler:
        logger.critical("Scaler не найден в артефактах! ОСТАНОВКА.")
        raise ValueError("Scaler не найден в артефактах.")

    try:
        X_scaled_np = scaler.transform(df_processed_for_model)
    except ValueError as e_scaler:
        logger.critical(f"Ошибка при scaler.transform: {e_scaler}. "
                        "Это обычно означает, что в df_processed_for_model остались нечисловые данные.", exc_info=True)
        # Логируем типы данных проблемных колонок
        for col in df_processed_for_model.columns:
            if not pd.api.types.is_numeric_dtype(df_processed_for_model[col]):
                logger.error(f"  Проблемная колонка '{col}' имеет тип {df_processed_for_model[col].dtype} и значение: {df_processed_for_model.loc[df.index[0], col]}")
        raise
        
    logger.debug(f"  Данные после масштабирования (форма): {X_scaled_np.shape}")
    logger.info("--- Предобработка входных данных ЗАВЕРШЕНА ---")
    return X_scaled_np, final_model_features


def make_prediction(movie_data_dict: Dict[str, Any]) -> float:
    logger.info(f"Получен запрос на предсказание (ключевые поля из movie_data_dict): "
                f"{ {k:v for k,v in movie_data_dict.items() if k in ['movie_title', 'director', 'budget', 'movie_year']} }")
    try:
        artifacts = get_artifacts() 
            
        processed_data_np, feature_names_for_dmatrix = preprocess_input_data(movie_data_dict, artifacts)
        
        log_dataframe_info(pd.DataFrame(processed_data_np, columns=feature_names_for_dmatrix), 
                              "processed_data_np (готово для DMatrix)", log_level=logging.INFO)
        
        model = artifacts.get('model') # Модель всегда должна быть
        if not model: # Эта проверка дублируется с get_artifacts, но для уверенности
            logger.critical("Модель не найдена в артефактах после get_artifacts! ОСТАНОВКА.")
            raise ValueError("Модель не найдена.")
        
        if processed_data_np.shape[1] != len(feature_names_for_dmatrix):
            msg = (f"Несоответствие признаков: DMatrix ожидает {len(feature_names_for_dmatrix)} ({feature_names_for_dmatrix[:3]}...), "
                   f"получено {processed_data_np.shape[1]}.")
            logger.error(msg); raise ValueError(msg)

        logger.info("Создание DMatrix и выполнение предсказания моделью...")
        dmatrix_processed = xgb.DMatrix(processed_data_np, feature_names=feature_names_for_dmatrix)
        prediction_value = model.predict(dmatrix_processed)
        logger.info(f"Предсказание моделью (сырое): {prediction_value}")
        
        if isinstance(prediction_value, np.ndarray) and prediction_value.ndim >= 1:
            final_prediction_scalar = float(prediction_value[0])
        else: 
            final_prediction_scalar = float(prediction_value)

        predicted_gross = round(final_prediction_scalar)
        logger.info(f"Финальное предсказание (округленное): ${predicted_gross:,.0f}")
        return predicted_gross
    except Exception as e:
        logger.error(f"Ошибка в процессе предсказания: {e}", exc_info=True)
        raise