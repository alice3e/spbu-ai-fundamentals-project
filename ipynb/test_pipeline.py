# test_pipeline.py
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import xgboost as xgb 
import logging

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, # INFO по умолчанию, можно изменить на DEBUG
                    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Вспомогательная функция для логирования DataFrame ---
def log_dataframe_info(df, name="DataFrame", max_rows=3, max_cols=10, log_level=logging.DEBUG):
    if not isinstance(df, pd.DataFrame): logger.log(log_level, f"'{name}' не DataFrame: {type(df)}"); return
    logger.log(log_level, f"Инфо о '{name}': Форма: {df.shape}")
    if df.empty: logger.log(log_level, f"  '{name}' пуст."); return
    if df.shape[1] <= max_cols: logger.log(log_level, f"  Первые {min(max_rows, df.shape[0])} строк:\n{df.head(max_rows).to_string()}")
    else: logger.log(log_level, f"  Первые {min(max_rows, df.shape[0])} строк (выборочно {max_cols} колонок):\n{df.iloc[:min(max_rows, df.shape[0]), list(range(max_cols//2)) + list(range(-max_cols//2, 0))].to_string()}")
    nan_counts = df.isnull().sum(); nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty: logger.log(log_level, f"  Колонки с NaN (количество):\n{nan_cols[nan_cols > 0].to_string()}")
    # else: logger.log(log_level, "  NaN в DataFrame отсутствуют.")


def convert_runtime_to_minutes(value):
    if pd.isna(value): return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def load_all_artifacts(artifacts_dir="saved_model_a5"):
    logger.info(f"Начало загрузки артефактов из директории: {artifacts_dir}")
    artifacts = {}
    files_map = {
        "model_booster_json": "movie_box_office_model.json", # Для XGBoost Booster
        "model_regressor_joblib": "movie_box_office_model.joblib", # Для XGBRegressor
        "numerical_imputer": "numerical_imputer.joblib",
        "categorical_imputer": "categorical_imputer.joblib",
        "onehot_encoder": "onehot_encoder.joblib",
        "target_encoder": "target_encoder.joblib",
        "scaler": "scaler.joblib",
        "column_info": "column_info.json"
    }
    try:
        for key, f_name in files_map.items():
            path = os.path.join(artifacts_dir, f_name)
            logger.debug(f"Загрузка: {key} из {path}")
            if not os.path.exists(path):
                if key in ["model_booster_json", "model_regressor_joblib"]: # Модель может быть в одном из форматов
                    logger.warning(f"Файл модели {f_name} не найден, будет использован другой формат, если доступен.")
                    artifacts[key] = None # Помечаем, что не загружен
                    continue
                logger.error(f"Файл не найден: {path}")
                raise FileNotFoundError(f"Не найден файл артефакта: {path}")
            
            if key == "model_booster_json" and os.path.exists(path): 
                model_obj = xgb.Booster()
                model_obj.load_model(path)
                artifacts[key] = model_obj
                logger.info(f"Модель XGBoost Booster '{f_name}' успешно загружена.")
            elif key == "model_regressor_joblib" and os.path.exists(path):
                artifacts[key] = joblib.load(path)
                logger.info(f"Модель XGBRegressor '{f_name}' успешно загружена.")
            elif f_name.endswith(".joblib"):
                artifacts[key] = joblib.load(path)
                logger.info(f"Артефакт '{key}' ({f_name}) успешно загружен.")
            elif f_name.endswith(".json") and key == "column_info":
                with open(path, "r", encoding='utf-8') as f:
                    artifacts[key] = json.load(f)
                logger.info(f"Артефакт '{key}' ({f_name}) успешно загружен.")
                logger.debug(f"Содержимое column_info.json (выборочно): "
                             f"features[:5]={artifacts[key].get('features',[])[:5]}, "
                             f"ohe_input_columns={artifacts[key].get('ohe_input_columns')}")
        
        # Выбираем, какую модель использовать (Booster приоритетнее, если есть JSON)
        if artifacts.get("model_booster_json"):
            artifacts["model"] = artifacts["model_booster_json"]
            logger.info("Используется модель XGBoost Booster (из .json).")
        elif artifacts.get("model_regressor_joblib"):
            artifacts["model"] = artifacts["model_regressor_joblib"]
            logger.info("Используется модель XGBRegressor (из .joblib).")
        else:
            logger.error("Ни одна из версий модели не была загружена.")
            raise ValueError("Не удалось загрузить основную модель.")
        
        # Удаляем лишние ключи, если они были None
        artifacts.pop("model_booster_json", None)
        artifacts.pop("model_regressor_joblib", None)

        logger.info("Все необходимые артефакты успешно загружены.")
        return artifacts
    except Exception as e:
        logger.critical(f"Критическая ошибка при загрузке артефактов: {e}", exc_info=True)
        raise RuntimeError(f"Не удалось загрузить артефакты: {e}")


def preprocess_dataframe(df_input: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    logger.info(f"--- НАЧАЛО ПРЕДОБРАБОТКИ DataFrame (форма: {df_input.shape}) ---")
    df = df_input.copy() # Работаем с копией
    log_dataframe_info(df, "df (начальный)")

    column_info = artifacts.get('column_info')
    if not column_info: logger.error("column_info отсутствует!"); raise ValueError("column_info не загружен.")

    # --- Шаг 0: Удаление начальных колонок ---
    initial_cols_to_drop = column_info.get("initial_columns_to_drop", [])
    if initial_cols_to_drop:
        cols_present_to_drop = [col for col in initial_cols_to_drop if col in df.columns]
        if cols_present_to_drop:
            df = df.drop(columns=cols_present_to_drop)
            logger.info(f"Удалены начальные колонки: {cols_present_to_drop}")
            log_dataframe_info(df, "df (после начального drop)")

    # --- Шаг 1: Feature Engineering (опыт) ---
    logger.info("Шаг 1: Feature Engineering (опыт)")
    exp_maps = column_info.get("experience_maps", {})
    personnel_cols_for_exp = column_info.get("personnel_cols_for_experience", [])
    for col in personnel_cols_for_exp:
        map_key = f"{col}_experience_map"
        if col in df.columns and map_key in exp_maps:
            df[f"{col}_experience"] = df[col].astype(str).map(exp_maps[map_key]).fillna(0)
            logger.debug(f"  Создан/обновлен признак {col}_experience.")
        elif col in df.columns: # Если колонка есть, но карты нет
            df[f"{col}_experience"] = 0
            logger.warning(f"  Карта для {map_key} не найдена, {col}_experience заполнено 0.")
        else:
            logger.warning(f"  Колонка {col} для опыта отсутствует во входных данных.")


    actor_exp_cols_generated = []
    actor_prefix = column_info.get("actor_cols_for_experience_prefix", "main_actor_")
    for i in range(1, 5):
        col_actor = f"{actor_prefix}{i}"
        map_key_actor = f"{col_actor}_experience_map"
        if col_actor in df.columns and map_key_actor in exp_maps:
            exp_col_name = f"{col_actor}_experience"
            df[exp_col_name] = df[col_actor].astype(str).map(exp_maps[map_key_actor]).fillna(0)
            actor_exp_cols_generated.append(exp_col_name)
            logger.debug(f"  Создан/обновлен признак {exp_col_name}.")
        elif col_actor in df.columns:
            df[f"{col_actor}_experience"] = 0 # Создаем колонку с нулями, если актер есть, но карты нет
            actor_exp_cols_generated.append(f"{col_actor}_experience")
            logger.warning(f"  Карта для {map_key_actor} не найдена, {col_actor}_experience заполнено 0.")
        else: # Если колонки актера нет, то и колонки опыта не будет
             logger.warning(f"  Колонка {col_actor} для опыта отсутствует.")
    
    if actor_exp_cols_generated:
        df["cast_popularity"] = df[actor_exp_cols_generated].sum(axis=1)
        logger.debug(f"  Создан/обновлен признак cast_popularity.")
    else:
        df["cast_popularity"] = 0
        logger.warning("  Колонки опыта актеров не сгенерированы, cast_popularity = 0.")
    log_dataframe_info(df, "df (после опыта)")

    # --- Шаг 2: Преобразование run_time ---
    logger.info("Шаг 2: Преобразование run_time")
    if 'run_time' in df.columns:
        df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes)
        logger.debug("  run_time преобразован в минуты.")
    log_dataframe_info(df, "df (после run_time)")

    # --- Шаг 3: Numerical Imputation ---
    logger.info("Шаг 3: Numerical Imputation")
    num_imputer = artifacts.get('numerical_imputer')
    if not num_imputer: logger.error("numerical_imputer не найден!"); raise ValueError("numerical_imputer не найден.")
    
    num_features_expected = column_info.get("numerical_imputer_features_in", [])
    if not num_features_expected:
        logger.warning("numerical_imputer_features_in не найден в column_info, используем атрибут импьютера.")
        num_features_expected = getattr(num_imputer, 'feature_names_in_', [])
        if isinstance(num_features_expected, np.ndarray): num_features_expected = num_features_expected.tolist()

    logger.debug(f"  Numerical Imputer будет применен к (ожидаемые): {num_features_expected}")
    
    # Создаем недостающие числовые колонки
    for col in num_features_expected:
        if col not in df.columns: df[col] = np.nan
            
    if num_features_expected: 
        # Важно: передавать в transform только те колонки, которые есть в df, НО в том порядке, как ожидает импьютер
        # Лучше передать DataFrame с ожидаемыми колонками, даже если некоторые из них пришлось создать с NaN
        cols_for_transform = [c for c in num_features_expected if c in df.columns] # Колонки, которые реально есть
        # Если не все ожидаемые колонки есть, это может быть проблемой, но мы уже создали недостающие с NaN
        if set(cols_for_transform) != set(num_features_expected):
             logger.warning(f"Не все ожидаемые колонки для num_imputer найдены в DataFrame. Ожидались: {num_features_expected}, найдены: {cols_for_transform}")
        
        df_subset_num = df[num_features_expected].copy() # Используем полный список ожидаемых для порядка
        log_dataframe_info(df_subset_num, "Данные для numerical_imputer (до transform)")
        imputed_values = num_imputer.transform(df_subset_num)
        imputed_df = pd.DataFrame(imputed_values, columns=num_features_expected, index=df.index)
        for col_name in num_features_expected: # Обновляем или добавляем колонки
            df[col_name] = imputed_df[col_name]
    else:
        logger.warning("  Список колонок для числовой импутации пуст.")
    log_dataframe_info(df, "df (после Numerical Imputation)")
    
    # --- Шаг 4: Grouped Imputation ---
    logger.info("Шаг 4: Grouped Imputation")
    group_maps = column_info.get("grouped_imputation_maps", {}) 
    group_impute_cols = column_info.get("cols_for_grouped_imputation_source_director", []) 
    logger.debug(f"  Колонки для групповой импутации по 'director': {group_impute_cols}")
    if 'director' in df.columns and group_impute_cols:
        for col_to_impute in group_impute_cols:
            if col_to_impute not in df.columns: df[col_to_impute] = np.nan # Создаем, если нет
            
            map_key = f"{col_to_impute}_director_mode_map"
            current_col_map = group_maps.get(map_key, {})
            
            # Создаем серию мод на основе df['director'] и карты
            # .astype(str) для director, т.к. ключи в карте - строки
            mapped_modes = df['director'].astype(str).map(current_col_map) 
            
            # Заполняем NaN в целевой колонке используя эту серию мод
            # Проверяем, что колонка существует перед fillna
            if col_to_impute in df.columns:
                df[col_to_impute].fillna(mapped_modes, inplace=True)
                logger.debug(f"    Пропуски в '{col_to_impute}' заполнены модами по 'director'.")
            else:
                logger.warning(f"    Колонка '{col_to_impute}' для групповой импутации отсутствует после создания с NaN.")
    log_dataframe_info(df, "df (после Grouped Imputation)")

    # --- Шаг 5: Fill 'Unknown' для специфических колонок ---
    logger.info("Шаг 5: Заполнение 'Unknown'")
    fill_unknown_cols = column_info.get("cols_to_fill_unknown_specific", []) 
    logger.debug(f"  Колонки для 'Unknown': {fill_unknown_cols}")
    if fill_unknown_cols:
        for col in fill_unknown_cols:
            if col not in df.columns: df[col] = np.nan
            # Заполняем 'Unknown', если значение NaN или пустая строка
            df[col] = df[col].apply(lambda x: 'Unknown' if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x)
            logger.debug(f"    '{col}' обработан для 'Unknown'.")
    log_dataframe_info(df, "df (после Fill Unknown)")

    # --- Шаг 6: Categorical Imputation (general) ---
    logger.info("Шаг 6: Categorical Imputation")
    cat_imputer = artifacts.get('categorical_imputer')
    if not cat_imputer: logger.error("categorical_imputer не найден!"); raise ValueError("categorical_imputer не найден.")
    
    cat_features_expected = column_info.get("categorical_imputer_features_in", [])
    if not cat_features_expected:
        logger.warning("categorical_imputer_features_in не найден в column_info, используем атрибут импьютера.")
        cat_features_expected = getattr(cat_imputer, 'feature_names_in_', [])
        if isinstance(cat_features_expected, np.ndarray): cat_features_expected = cat_features_expected.tolist()
    logger.debug(f"  Categorical Imputer будет применен к (ожидаемые): {cat_features_expected}")

    for col in cat_features_expected: # Создаем недостающие
        if col not in df.columns: df[col] = np.nan
            
    if cat_features_expected:
        cols_for_cat_transform = cat_features_expected
        df_subset_cat = df[cols_for_cat_transform].copy() # Порядок важен
        log_dataframe_info(df_subset_cat, "Данные для categorical_imputer (до transform)")
        imputed_cat_values = cat_imputer.transform(df_subset_cat)
        imputed_df_cat = pd.DataFrame(imputed_cat_values, columns=cols_for_cat_transform, index=df.index)
        for col_name_cat in cols_for_cat_transform:
            df[col_name_cat] = imputed_df_cat[col_name_cat]
    else:
        logger.warning("  Список колонок для категориальной импутации пуст.")
    log_dataframe_info(df, "df (после Categorical Imputation)")

    # --- Шаг 7: One-Hot Encoding ---
    logger.info("Шаг 7: One-Hot Encoding")
    ohe = artifacts.get('onehot_encoder')
    if not ohe: logger.error("onehot_encoder не найден!"); raise ValueError("onehot_encoder не найден.")

    ohe_input_columns = column_info.get("ohe_input_columns", [])
    if not ohe_input_columns:
        if hasattr(ohe, 'feature_names_in_'): ohe_input_columns = ohe.feature_names_in_.tolist()
        else: logger.error("ohe_input_columns не найдены!"); raise ValueError("ohe_input_columns не найдены.")
    logger.debug(f"  Исходные колонки для OHE: {ohe_input_columns}")

    for col in ohe_input_columns: # Убедимся, что колонки есть и имеют строковый тип
        if col not in df.columns: df[col] = "Unknown"; logger.warning(f"  OHE колонка '{col}' отсутствовала, заполнена 'Unknown'.")
        elif df[col].isnull().any(): # Если есть NaN в колонке
            df[col] = df[col].fillna("Unknown").astype(str) # Заполняем NaN и приводим к строке
            logger.debug(f"  В OHE колонке '{col}' NaN заменены на 'Unknown' и тип изменен на str.")
        elif df[col].dtype != 'object' and df[col].dtype != 'string':
             df[col] = df[col].astype(str)
    
    if ohe_input_columns: 
        log_dataframe_info(df[ohe_input_columns], f"Данные для OHE (колонки {ohe_input_columns})")
        ohe_transformed_data = ohe.transform(df[ohe_input_columns])
        
        ohe_feature_names_arr = np.array([])
        if hasattr(ohe, 'get_feature_names_out'): ohe_feature_names_arr = ohe.get_feature_names_out(ohe_input_columns)
        elif hasattr(ohe, 'get_feature_names'): ohe_feature_names_arr = ohe.get_feature_names(ohe_input_columns)
        
        if not ohe_feature_names_arr.size > 0 : logger.error("Не удалось получить имена OHE!"); raise RuntimeError("Имена OHE пусты.")
        ohe_feature_names = list(ohe_feature_names_arr)
        logger.debug(f"  OHE сгенерировал {len(ohe_feature_names)} колонок: {ohe_feature_names[:3]}...{ohe_feature_names[-3:]}")

        if hasattr(ohe_transformed_data, "toarray"): ohe_df = pd.DataFrame(ohe_transformed_data.toarray(), columns=ohe_feature_names, index=df.index)
        else: ohe_df = pd.DataFrame(ohe_transformed_data, columns=ohe_feature_names, index=df.index)
        
        df = df.drop(columns=ohe_input_columns, errors='ignore')
        df = pd.concat([df, ohe_df], axis=1)
    log_dataframe_info(df, "df (после OHE)")

    # --- Шаг 8: Target Encoding ---
    logger.info("Шаг 8: Target Encoding")
    target_encoder = artifacts.get('target_encoder')
    if not target_encoder: logger.error("target_encoder не найден!"); raise ValueError("target_encoder не найден.")
    
    te_input_columns = column_info.get("target_encoding_columns", [])
    if not te_input_columns:
        if hasattr(target_encoder, 'cols'): te_input_columns = target_encoder.cols
        else: logger.error("target_encoding_columns не найдены!"); raise ValueError("target_encoding_columns не найдены.")
    logger.debug(f"  Исходные колонки для Target Encoding: {te_input_columns}")

    cols_for_te_transform = []
    for col in te_input_columns: # Убедимся, что колонки есть и имеют строковый тип
        if col not in df.columns:
            # TE из category_encoders обычно имеет handle_unknown и handle_missing
            # Если колонки нет, но она была в списке cols при fit, он может выдать ошибку.
            # Либо создаем с 'Unknown', либо полагаемся на handle_unknown='value' (если он так настроен)
            df[col] = "Unknown" # Безопасный вариант, если TE не обучен на NaN
            logger.warning(f"  TE колонка '{col}' отсутствовала, заполнена 'Unknown'.")
            cols_for_te_transform.append(col)
        elif df[col].isnull().any():
            df[col] = df[col].fillna("Unknown").astype(str)
            logger.debug(f"  В TE колонке '{col}' NaN заменены на 'Unknown' и тип изменен на str.")
            cols_for_te_transform.append(col)
        elif df[col].dtype != 'object' and df[col].dtype != 'string':
             df[col] = df[col].astype(str)
             cols_for_te_transform.append(col)
        else: # Колонка есть и уже строковая
            cols_for_te_transform.append(col)
            
    if target_encoder and cols_for_te_transform:
        log_dataframe_info(df[cols_for_te_transform], f"Данные для Target Encoder (колонки {cols_for_te_transform})")
        transformed_te_vals = target_encoder.transform(df[cols_for_te_transform])
        # transformed_te_vals будет DataFrame с теми же именами колонок
        df[cols_for_te_transform] = transformed_te_vals
    log_dataframe_info(df, "df (после Target Encoding)")
    
    # --- Шаг 9: Reindex ---
    logger.info("Шаг 9: Reindex")
    final_model_features = column_info.get("features") 
    if not isinstance(final_model_features, list) or not final_model_features:
        logger.error("'features' из column_info некорректен!"); raise ValueError("'features' должен быть непустым списком.")
    logger.debug(f"  Ожидаемые финальные признаки (всего {len(final_model_features)}): {final_model_features[:3]}...{final_model_features[-3:]}")
    
    # Создаем DataFrame с нужными колонками и порядком, заполняем 0
    df_processed_for_model = pd.DataFrame(0, index=df.index, columns=final_model_features)
    
    # Заполняем значениями из df, где колонки совпадают
    common_cols = df_processed_for_model.columns.intersection(df.columns)
    logger.debug(f"  Общие колонки для переноса в финальный DataFrame (всего {len(common_cols)}): {list(common_cols)[:3]}...")
    if not common_cols.empty:
        df_processed_for_model[common_cols] = df[common_cols]

    missing_in_df = set(final_model_features) - set(df.columns)
    if missing_in_df: logger.warning(f"  {len(missing_in_df)} final_features отсутствовали в df и будут 0: {list(missing_in_df)[:3]}...")
    extra_in_df = set(df.columns) - set(final_model_features)
    if extra_in_df: logger.warning(f"  {len(extra_in_df)} колонок из df не являются final_features: {list(extra_in_df)[:3]}...")
    
    if df_processed_for_model.isnull().any().any():
        nan_cols_final = df_processed_for_model.columns[df_processed_for_model.isnull().any()].tolist()
        logger.warning(f"  NaN обнаружены в df_processed_for_model перед масштабированием: {nan_cols_final}. Заполнение нулями.")
        df_processed_for_model.fillna(0, inplace=True)
    log_dataframe_info(df_processed_for_model, "df_processed_for_model (после Reindex, перед Scaling)")
    
    # --- Шаг 10: Scaling ---
    logger.info("Шаг 10: Scaling")
    scaler = artifacts.get('scaler')
    if not scaler: logger.error("scaler не найден!"); raise ValueError("scaler не найден.")
    X_scaled_np = scaler.transform(df_processed_for_model) # Результат - NumPy array
    logger.debug(f"  Данные после масштабирования (форма): {X_scaled_np.shape}")
    logger.info("--- Предобработка DataFrame ЗАВЕРШЕНА ---")
    return X_scaled_np, final_model_features # Возвращаем NumPy array и имена колонок


def apply_predictions(input_csv_path: str, output_csv_path: str, artifacts_dir: str):
    logger.info(f"Начало процесса предсказания для файла: {input_csv_path}")
    
    artifacts = load_all_artifacts(artifacts_dir)
    if artifacts is None:
        logger.error("Не удалось загрузить артефакты. Прерывание.")
        return

    try:
        df_test_raw = pd.read_csv(input_csv_path)
        logger.info(f"Входной CSV файл '{input_csv_path}' успешно прочитан. Строк: {len(df_test_raw)}")
        log_dataframe_info(df_test_raw, "df_test_raw (исходный)")
    except FileNotFoundError:
        logger.error(f"Входной файл CSV не найден: {input_csv_path}")
        return
    except Exception as e:
        logger.error(f"Ошибка при чтении CSV файла {input_csv_path}: {e}", exc_info=True)
        return

    # Сохраняем копию для итогового файла, чтобы не потерять оригинальные колонки
    df_output = df_test_raw.copy()

    # Предобработка всего DataFrame
    try:
        processed_data_np, feature_names_for_dmatrix = preprocess_dataframe(df_test_raw, artifacts)
    except Exception as e:
        logger.error(f"Ошибка на этапе предобработки данных: {e}", exc_info=True)
        return # Прерываем, если предобработка не удалась

    # Предсказание
    model = artifacts.get('model')
    if not model: logger.error("Модель не найдена в артефактах!"); raise ValueError("Модель не найдена.")

    logger.info(f"Выполнение предсказаний для {processed_data_np.shape[0]} строк...")
    try:
        dmatrix_processed = xgb.DMatrix(processed_data_np, feature_names=feature_names_for_dmatrix)
        predictions_raw = model.predict(dmatrix_processed)
        logger.info(f"Предсказания получены (форма сырых предсказаний: {predictions_raw.shape}).")
    except Exception as e:
        logger.error(f"Ошибка при выполнении model.predict(): {e}", exc_info=True)
        return

    # Добавляем предсказания к исходному DataFrame (или его копии)
    # Округляем предсказания
    df_output['predicted_worldwide'] = np.round(predictions_raw).astype(float) # float для возможности NaN, int если уверены
    log_dataframe_info(df_output[['movie_title', 'worldwide', 'predicted_worldwide'] if 'worldwide' in df_output else ['movie_title', 'predicted_worldwide']], 
                       "df_output (с предсказаниями)", log_level=logging.INFO)

    # Сохранение результата
    try:
        df_output.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"Результаты с предсказаниями сохранены в файл: {output_csv_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла с предсказаниями {output_csv_path}: {e}", exc_info=True)