# predictor_test.py
import pandas as pd
import numpy as np
import argparse
import os
import sys # Для добавления пути к predictor
import logging

# --- Добавляем путь к директории model_service, чтобы импортировать predictor ---
# Это нужно, если predictor_test.py находится не в model_service/
# Предположим, что predictor_test.py находится в корне проекта,
# а predictor.py в model_service/
current_dir = os.path.dirname(os.path.abspath(__file__))
model_service_dir = os.path.join(current_dir, "model_service") # Путь к model_service/
if model_service_dir not in sys.path:
    sys.path.insert(0, model_service_dir)

try:
    import predictor # Теперь импортируем наш predictor.py из model_service
    print(f"Модуль predictor успешно импортирован из: {predictor.__file__}")
except ImportError as e:
    print(f"Ошибка: Не удалось импортировать модуль predictor. {e}")
    print(f"Убедитесь, что predictor.py находится в {model_service_dir} или sys.path настроен правильно.")
    sys.exit(1)
except Exception as e:
    print(f"Неожиданная ошибка при импорте predictor: {e}")
    sys.exit(1)

# Настроим логгер из predictor, чтобы видеть его вывод
predictor.logger.setLevel(logging.INFO) # Установим INFO, можно DEBUG для большей детализации

def run_predictor_comparison(input_csv_path: str, artifacts_sub_dir: str = "saved_model"):
    """
    Читает CSV, делает предсказания для каждой строки с помощью predictor.py
    и сравнивает с существующими предсказаниями в файле.
    """
    if not os.path.exists(input_csv_path):
        print(f"Ошибка: Входной файл не найден: {input_csv_path}")
        return

    # Устанавливаем MODEL_DIR для модуля predictor
    # predictor.MODEL_DIR должен указывать на директорию относительно predictor.py
    # Если predictor.py находится в model_service/, а saved_model/ внутри model_service/,
    # то MODEL_DIR = "saved_model" в predictor.py - это правильно.
    # Здесь мы просто убедимся, что он знает, где искать артефакты относительно своего местоположения.
    # В данном случае, мы не меняем MODEL_DIR в predictor.py, он сам должен знать свой путь.
    # Мы передаем artifacts_dir в get_artifacts, если бы это было нужно.
    # Но get_artifacts в текущей версии использует свой MODEL_DIR.

    print(f"Используется MODEL_DIR в predictor: {predictor.MODEL_DIR}")
    # Если MODEL_DIR в predictor.py относительный ("saved_model"), то
    # saved_model должна быть в model_service/saved_model/

    df_test = pd.read_csv(input_csv_path)
    print(f"Прочитан файл '{input_csv_path}', строк: {len(df_test)}")

    if 'predicted_worldwide' not in df_test.columns:
        print(f"Ошибка: Колонка 'predicted_worldwide' для сравнения отсутствует в файле {input_csv_path}")
        return
    
    # Колонки, которые мы ожидаем передать в predictor (согласно Pydantic модели MovieDataInput)
    # Эти колонки должны быть в вашем output_test.csv
    expected_input_cols_for_predictor = [
        "movie_title", "movie_year", "budget", "run_time", "mpaa",
        "director", "writer", "producer", "composer", "cinematographer", "distributor",
        "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4",
        "genre_1", "genre_2", "genre_3", "genre_4"
    ]
    
    # Проверка наличия всех ожидаемых колонок в CSV
    missing_cols = [col for col in expected_input_cols_for_predictor if col not in df_test.columns]
    if missing_cols:
        print(f"Ошибка: В файле {input_csv_path} отсутствуют необходимые колонки: {missing_cols}")
        return

    results = []
    total_rows = len(df_test)
    
    # Загружаем артефакты ОДИН РАЗ перед циклом
    try:
        print("Предварительная загрузка артефактов...")
        predictor_artifacts = predictor.get_artifacts() # Это загрузит и закэширует
        if predictor_artifacts is None:
            print("Не удалось загрузить артефакты. Тестирование прервано.")
            return
        print("Артефакты для тестирования загружены.")
    except Exception as e:
        print(f"Критическая ошибка при загрузке артефактов для теста: {e}")
        return

    for index, row in df_test.iterrows():
        print(f"\n--- Обработка строки {index + 1}/{total_rows}: {row.get('movie_title', 'Без названия')} ---")
        
        movie_data_dict_input = {}
        for col in expected_input_cols_for_predictor:
            value = row.get(col)
            # Преобразуем пустые строки/NaN в None для консистентности с FastAPI/Pydantic
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                movie_data_dict_input[col] = None
            else:
                movie_data_dict_input[col] = value
        
        # Явное преобразование числовых полей (Pydantic это делает, но здесь повторим)
        for num_field in ["budget", "movie_year"]:
            val = movie_data_dict_input.get(num_field)
            if val is not None:
                try: movie_data_dict_input[num_field] = float(val)
                except (ValueError, TypeError): movie_data_dict_input[num_field] = np.nan
            else: movie_data_dict_input[num_field] = np.nan


        reference_prediction = row['predicted_worldwide']
        
        try:
            # Вызываем функцию предсказания из модуля predictor
            # Передаем movie_data_dict_input, который имитирует JSON от FastAPI
            current_prediction = predictor.make_prediction(movie_data_dict_input)
            
            match = False
            if pd.notna(reference_prediction) and pd.notna(current_prediction):
                # Сравнение с небольшой погрешностью для float
                if abs(float(current_prediction) - float(reference_prediction)) < 1: # Сравниваем как float
                    match = True
            elif pd.isna(reference_prediction) and pd.isna(current_prediction):
                match = True # Оба NaN считаем совпадением

            results.append({
                "movie_title": row.get('movie_title', 'N/A'),
                "reference_prediction": reference_prediction,
                "predictor_prediction": current_prediction,
                "match": match,
                "difference": abs(float(current_prediction) - float(reference_prediction)) if pd.notna(reference_prediction) and pd.notna(current_prediction) else np.nan
            })
            
            if match:
                print(f"  OK: Предсказание совпало. Ref: {reference_prediction}, Pred: {current_prediction}")
            else:
                print(f"  MISMATCH: Предсказание НЕ совпало! Ref: {reference_prediction}, Pred: {current_prediction}, Diff: {results[-1]['difference']}")

        except Exception as e:
            print(f"  ОШИБКА при предсказании для фильма '{row.get('movie_title', 'N/A')}': {e}")
            results.append({
                "movie_title": row.get('movie_title', 'N/A'),
                "reference_prediction": reference_prediction,
                "predictor_prediction": "ERROR",
                "match": False,
                "difference": np.nan
            })

    # Вывод итоговой статистики
    print("\n--- Итоги тестирования predictor.py ---")
    if results:
        df_results = pd.DataFrame(results)
        total_tests = len(df_results)
        matches = df_results["match"].sum()
        mismatches = total_tests - matches
        
        print(f"Всего тестов: {total_tests}")
        print(f"Совпадений: {matches} ({matches/total_tests*100:.2f}%)")
        print(f"Расхождений: {mismatches} ({mismatches/total_tests*100:.2f}%)")

        if mismatches > 0:
            print("\nФильмы с расхождениями:")
            print(df_results[~df_results["match"]].to_string())
        
        # Статистика по разнице
        valid_diffs = df_results['difference'].dropna()
        if not valid_diffs.empty:
            print("\nСтатистика по абсолютной разнице (для несовпавших, где оба предсказания числа):")
            print(f"  Средняя разница: {valid_diffs[df_results['match']==False].mean():,.0f}")
            print(f"  Медианная разница: {valid_diffs[df_results['match']==False].median():,.0f}")
            print(f"  Макс. разница: {valid_diffs[df_results['match']==False].max():,.0f}")
            print(f"  Мин. разница: {valid_diffs[df_results['match']==False].min():,.0f}")

    else:
        print("Тесты не были выполнены.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирует predictor.py на данных из CSV файла, сравнивая с существующими предсказаниями.")
    parser.add_argument("input_comparison_csv", 
                        help="Путь к CSV файлу (например, output_test.csv), содержащему оригинальные данные и колонку 'predicted_worldwide' для сравнения.")
    parser.add_argument("--artifacts_base_dir", 
                        default=os.path.join(current_dir, "model_service"), 
                        help="Базовая директория, относительно которой predictor.py ищет папку 'saved_model' (по умолчанию: ./model_service).")
    parser.add_argument("--log_level", 
                        default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Уровень логирования для модуля predictor.")
    
    args = parser.parse_args()

    # Устанавливаем уровень логирования для модуля predictor
    # Имя логгера в predictor.py: logging.getLogger(__name__) -> "predictor"
    predictor_logger_instance = logging.getLogger("predictor") 
    predictor_logger_instance.setLevel(args.log_level.upper())
    
    # Если predictor.py находится НЕ в model_service, а в корне, то artifacts_base_dir будет current_dir
    # Важно, чтобы путь к predictor.MODEL_DIR был правильным.
    # predictor.MODEL_DIR в predictor.py должен быть "saved_model".
    # Мы предполагаем, что скрипт predictor_test.py запускается из корневой директории,
    # а predictor.py находится в model_service/ и его MODEL_DIR указывает на model_service/saved_model/
    # Поэтому get_artifacts() в predictor.py сам найдет saved_model/ относительно своего местоположения.
    # Аргумент artifacts_base_dir здесь больше для информации или если бы мы хотели менять MODEL_DIR в predictor динамически.
    
    print(f"Тестирование на файле: {args.input_comparison_csv}")
    print(f"Базовая директория для поиска артефактов модулем predictor: {os.path.abspath(args.artifacts_base_dir)}")
    print(f"Уровень логирования для predictor: {args.log_level.upper()}")

    run_predictor_comparison(args.input_comparison_csv)
    