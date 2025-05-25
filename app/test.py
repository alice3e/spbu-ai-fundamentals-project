# test.py
import pandas as pd
import numpy as np
import joblib
import json
import re 
import os
import random # Для выбора случайных значений из списков опций

# --- Глобальные переменные и константы ---
MODEL_DIR = "saved_model" # Директория, где хранятся артефакты

# --- Импорт доступных опций ---
try:
    import available_options as opts
    print("Файл available_options.py успешно импортирован.")
except ImportError:
    print("Ошибка: Файл available_options.py не найден. Пожалуйста, запустите generate_options.py.")
    # Создаем заглушки, чтобы скрипт мог продолжать работу с ограниченной функциональностью
    # или можно просто завершить выполнение: exit()
    class opts: # type: ignore
        ALL_DIRECTORS = ["Default Director"]
        ALL_WRITERS = ["Default Writer"]
        ALL_PRODUCERS = ["Default Producer"]
        ALL_COMPOSERS = ["Default Composer"]
        ALL_CINEMATOGRAPHERS = ["Default Cinematographer"]
        ALL_ACTORS = ["Default Actor 1", "Default Actor 2"]
        ALL_MPAA_RATINGS = ["PG-13"]
        ALL_GENRES = ["Action", "Adventure"]
        ALL_DISTRIBUTORS = ["Default Distributor"]
    print("Используются значения по умолчанию для опций.")


# --- Вспомогательные функции (должны быть идентичны тем, что использовались при обучении) ---
def convert_runtime_to_minutes(value):
    if pd.isna(value): return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def load_artifacts(model_dir):
    print(f"\nЗагрузка артефактов из директории: {model_dir}")
    artifacts = {}
    required_files = [
        "movie_box_office_model.joblib", "numerical_imputer.joblib",
        "categorical_imputer.joblib", "target_encoder.joblib",
        "scaler.joblib", "column_info.json"
    ]
    try:
        for f_name in required_files:
            path = os.path.join(model_dir, f_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Не найден обязательный файл артефакта: {path}")
            
            if f_name.endswith(".joblib"):
                artifacts[f_name.split('.')[0]] = joblib.load(path)
            elif f_name.endswith(".json"):
                with open(path, "r", encoding='utf-8') as f:
                    artifacts[f_name.split('.')[0]] = json.load(f)
        
        # Переименуем для удобства (если имена файлов стандартные)
        artifacts['model'] = artifacts.pop('movie_box_office_model', None)
        artifacts['numerical_imputer'] = artifacts.pop('numerical_imputer', None)
        artifacts['categorical_imputer'] = artifacts.pop('categorical_imputer', None)
        artifacts['target_encoder'] = artifacts.pop('target_encoder', None)
        artifacts['scaler'] = artifacts.pop('scaler', None)
        artifacts['column_info'] = artifacts.pop('column_info', None)

        # Проверка, что все ключи загрузились
        for key in ['model', 'numerical_imputer', 'categorical_imputer', 'target_encoder', 'scaler', 'column_info']:
            if artifacts.get(key) is None:
                raise ValueError(f"Артефакт '{key}' не был корректно загружен или переименован.")

        print("Все артефакты успешно загружены.")
        if hasattr(artifacts['numerical_imputer'], 'feature_names_in_'):
            print(f"  Numerical Imputer был обучен на: {artifacts['numerical_imputer'].feature_names_in_}")
        if hasattr(artifacts['categorical_imputer'], 'feature_names_in_'):
            print(f"  Categorical Imputer был обучен на: {artifacts['categorical_imputer'].feature_names_in_}")
        if hasattr(artifacts['target_encoder'], 'cols'):
            print(f"  Target Encoder был обучен на: {artifacts['target_encoder'].cols}")
        return artifacts
    except Exception as e:
        print(f"Произошла ошибка при загрузке артефактов: {e}")
        return None

def preprocess_single_movie_data(movie_data_dict, artifacts):
    # print("\nНачало предобработки данных для фильма...")
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
    num_features_expected_arr = getattr(artifacts['numerical_imputer'], 'feature_names_in_', np.array([])) # Получаем как массив
    num_features_expected = list(num_features_expected_arr) # Преобразуем в список для удобства

    for col in num_features_expected:
        if col not in df_movie.columns: df_movie[col] = np.nan
    
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: Проверяем длину списка (или массива)
    if len(num_features_expected) > 0: # <--- ИЗМЕНЕНО
        # Убедимся, что передаем DataFrame с колонками в том же порядке, что и при fit
        df_movie_subset_for_imputer = df_movie[num_features_expected].copy()
        imputed_num_vals = artifacts['numerical_imputer'].transform(df_movie_subset_for_imputer)
        for i, col_name in enumerate(num_features_expected):
            df_movie[col_name] = imputed_num_vals[0, i]
    
    # 4. Grouped Imputation
    group_maps = column_info.get("grouped_imputation_maps", {})
    group_impute_cols = column_info.get("group_impute_director_dependent_cols", [])
    director_name_str = str(df_movie.loc[0, 'director']) if 'director' in df_movie.columns and pd.notna(df_movie.loc[0, 'director']) else None
    if director_name_str:
        for col in group_impute_cols:
            if col not in df_movie.columns: df_movie[col] = np.nan
            if pd.isna(df_movie.loc[0, col]):
                map_key = f"{col}_director_mode_map"
                mode_val = group_maps.get(map_key, {}).get(director_name_str)
                if mode_val is not None: df_movie.loc[0, col] = mode_val
    
    # 5. Fill 'Unknown'
    fill_unknown_cols = column_info.get("fill_unknown_cols", [])
    for col in fill_unknown_cols:
        if col not in df_movie.columns: df_movie[col] = np.nan
        if pd.isna(df_movie.loc[0, col]): df_movie.loc[0, col] = 'Unknown'

    # 6. Categorical Imputation
    cat_features_expected_arr = getattr(artifacts['categorical_imputer'], 'feature_names_in_', np.array([])) # Получаем как массив
    cat_features_expected = list(cat_features_expected_arr) # Преобразуем в список

    for col in cat_features_expected:
        if col not in df_movie.columns: df_movie[col] = np.nan
    
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: Проверяем длину списка (или массива)
    if len(cat_features_expected) > 0: # <--- ИЗМЕНЕНО
        df_movie_subset_for_cat_imputer = df_movie[cat_features_expected].copy()
        imputed_cat_vals = artifacts['categorical_imputer'].transform(df_movie_subset_for_cat_imputer)
        for i, col_name in enumerate(cat_features_expected):
            df_movie[col_name] = imputed_cat_vals[0, i]

    # 7. One-Hot Encoding
    ohe_cols = column_info.get("one_hot_encoded_cols", [])
    actual_ohe_cols = [col for col in ohe_cols if col in df_movie.columns]
    if actual_ohe_cols:
        df_movie = pd.get_dummies(df_movie, columns=actual_ohe_cols, dummy_na=False)

    # 8. Target Encoding
    te_cols = column_info.get("target_encoded_cols", [])
    # actual_te_cols = [col for col in te_cols if col in df_movie.columns] # Эта строка не нужна, используем te_encoder_fit_cols
    
    te_encoder_fit_cols = getattr(artifacts['target_encoder'], 'cols', []) # Колонки, на которых TE был обучен
    cols_for_te_transform = [col for col in te_encoder_fit_cols if col in df_movie.columns] # Которые есть в текущем df

    if artifacts['target_encoder'] and cols_for_te_transform:
        transformed_te_vals = artifacts['target_encoder'].transform(df_movie[cols_for_te_transform])
        if isinstance(transformed_te_vals, pd.Series):
             df_movie[cols_for_te_transform[0]] = transformed_te_vals.values[0] 
        elif isinstance(transformed_te_vals, pd.DataFrame):
             df_movie[cols_for_te_transform] = transformed_te_vals.values
    
    # 9. Reindex
    final_features = column_info.get("final_model_features", [])
    if not final_features: raise ValueError("final_model_features не найдены в column_info.")
    
    df_processed_for_model = pd.DataFrame(0, index=[0], columns=final_features)
    
    for col in df_movie.columns:
        if col in df_processed_for_model.columns:
            df_processed_for_model[col] = df_movie[col].values

    if df_processed_for_model.isnull().any().any():
        # print("  Предупреждение: NaN обнаружены перед масштабированием. Заполнение нулями.")
        df_processed_for_model.fillna(0, inplace=True)
    
    # 10. Scaling
    X_scaled = artifacts['scaler'].transform(df_processed_for_model)
    # print("  Предобработка фильма завершена.")
    return X_scaled

def predict_movie_gross(movie_data_dict, artifacts):
    if artifacts is None: return None
    try:
        processed_data = preprocess_single_movie_data(movie_data_dict, artifacts)
        prediction = artifacts['model'].predict(processed_data)
        return prediction[0]
    except Exception as e:
        print(f"Ошибка во время предсказания: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_random_movie_sample(options_module):
    """Генерирует словарь с данными для случайного фильма, используя списки из opts."""
    
    # Безопасный выбор из списка, возвращает None если список пуст
    def safe_choice(lst, default=None):
        return random.choice(lst) if lst else default

    def safe_sample(lst, k, default_fill=pd.NA):
        k_actual = min(k, len(lst))
        sampled = random.sample(lst, k_actual) if lst else []
        # Дополняем до k элементов, если необходимо
        return sampled + [default_fill] * (k - k_actual)

    movie_sample = {
        'movie_title': f"The Avengers",
        'distributor': safe_choice(options_module.ALL_DISTRIBUTORS, "Unknown Distributor"),
        'budget': random.randint(1000000, 200000000), # Широкий диапазон бюджета
        'run_time': f"{random.randint(1,3)} hr {random.randint(0,59)} min",
        'mpaa': safe_choice(options_module.ALL_MPAA_RATINGS, "PG-13"),
        'director': safe_choice(options_module.ALL_DIRECTORS, "John Doe"),
        'writer': safe_choice(options_module.ALL_WRITERS, "Jane Script"),
        'producer': safe_choice(options_module.ALL_PRODUCERS, "Big Budget Productions"),
        'composer': safe_choice(options_module.ALL_COMPOSERS, "Orchestra Max"),
        'cinematographer': safe_choice(options_module.ALL_CINEMATOGRAPHERS, "Steady Cam"),
        'movie_year': random.randint(1980, 2025), # Добавляем movie_year, т.к. он часто нужен
    }

    # Актеры
    actors = safe_sample(options_module.ALL_ACTORS, 4)
    for i in range(1, 5):
        movie_sample[f'main_actor_{i}'] = actors[i-1]

    # Жанры
    genres = safe_sample(options_module.ALL_GENRES, 4, default_fill=np.nan) # NaN для отсутствующих жанров
    for i in range(1, 5):
        movie_sample[f'genre_{i}'] = genres[i-1]
        
    # Добавим другие числовые поля, которые мог ожидать numerical_imputer, если они известны
    # Например, если импьютер обучался на 'screens', 'opening_weekend' и т.д.
    # Для примера, оставим только 'budget' и 'movie_year' как основные числовые.
    # Если column_info.json содержит num_cols_imputed_on_train, можно их использовать.

    return movie_sample


# --- Точка входа ---
if __name__ == "__main__":
    artifacts = load_artifacts(MODEL_DIR)

    if artifacts:
        print("\n--- Предсказание для случайно сгенерированного фильма ---")
        random_movie = generate_random_movie_sample(opts)
        
        print("Сгенерированные данные для фильма:")
        for key, value in random_movie.items():
            print(f"  {key}: {value}")

        predicted_gross_random = predict_movie_gross(random_movie, artifacts)

        if predicted_gross_random is not None:
            print(f"\nПредсказание для фильма '{random_movie.get('movie_title', 'N/A')}':")
            print(f"  Предсказанные мировые сборы: ${predicted_gross_random:,.2f}")
        else:
            print(f"Не удалось получить предсказание для фильма '{random_movie.get('movie_title', 'N/A')}'.")

        print("\n--- Тестирование на данных из файла test.csv (если он существует) ---")
        test_csv_path = "test.csv" 
        if os.path.exists(test_csv_path):
            try:
                test_df_from_file = pd.read_csv(test_csv_path)
                print(f"Загружен {test_csv_path} ({len(test_df_from_file)} фильмов). Обработка первых N фильмов...")
                
                N_FILMS_TO_PREDICT = min(3, len(test_df_from_file)) # Обрабатываем до 3 фильмов
                
                for index, row in test_df_from_file.head(N_FILMS_TO_PREDICT).iterrows():
                    print(f"\nОбработка фильма #{index+1} из test.csv: {row.get('movie_title', 'Без названия')}")
                    movie_dict_from_file = row.to_dict()
                    
                    target_var_name = artifacts['column_info'].get("target_variable_name", "worldwide")
                    true_gross_for_this_movie = movie_dict_from_file.pop(target_var_name, None)
                    
                    cols_to_drop_on_load = artifacts['column_info'].get("columns_to_drop_on_load", [])
                    original_title_from_row = row.get('movie_title', f"Фильм #{index+1}") # Сохраняем до возможного удаления
                    for col_d in cols_to_drop_on_load:
                        movie_dict_from_file.pop(col_d, None)
                    
                    predicted_gross_file = predict_movie_gross(movie_dict_from_file, artifacts)
                    
                    if predicted_gross_file is not None:
                        print(f"  Фильм: {original_title_from_row}")
                        print(f"    Предсказанные сборы: ${predicted_gross_file:,.2f}")
                        if true_gross_for_this_movie is not None:
                            print(f"    Фактические сборы: ${float(true_gross_for_this_movie):,.2f}") # Преобразуем в float для форматирования
                    else:
                        print(f"  Не удалось получить предсказание для фильма: {original_title_from_row}")
            except Exception as e:
                print(f"Ошибка при обработке {test_csv_path}: {e}")
        else:
            print(f"Файл {test_csv_path} не найден. Пропуск пакетного предсказания.")
    else:
        print("Не удалось загрузить артефакты. Предсказание невозможно.")