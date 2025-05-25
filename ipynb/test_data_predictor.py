# test_data_predictor.py
import argparse
import test_pipeline # Импортируем модуль test_pipeline.py с основной логикой
import logging
import os # Добавлен импорт os для проверки существования файла

# --- Настройка базового логирования для этого скрипта ---
# Логгер из test_pipeline будет иметь свои настройки, но мы можем
# настроить корневой логгер или логгер самого test_pipeline, если нужно.
# test_pipeline.py уже настраивает свой логгер, так что здесь
# можно просто убедиться, что он активен, или переопределить уровень.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сделать предсказания кассовых сборов для фильмов из CSV файла, используя test_pipeline."
    )
    parser.add_argument(
        "input_csv", 
        default="test.csv", 
        nargs='?', # Делаем аргумент опциональным, чтобы использовать default
        help="Путь к входному CSV файлу с данными фильмов (по умолчанию: test.csv)."
    )
    parser.add_argument(
        "output_csv", 
        default="test_predictions.csv", 
        nargs='?', # Делаем аргумент опциональным
        help="Путь для сохранения CSV файла с предсказаниями (по умолчанию: test_predictions.csv)."
    )
    parser.add_argument(
        "--artifacts_dir", 
        default="saved_model_a5", 
        help="Директория с сохраненными артефактами модели (по умолчанию: saved_model)."
    )
    parser.add_argument(
        "--log_level", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Уровень логирования для модуля test_pipeline (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )
    
    args = parser.parse_args()

    # Устанавливаем уровень логирования для логгера из test_pipeline
    # test_pipeline.logger - это logging.getLogger(__name__) в test_pipeline.py,
    # что означает, что его имя будет "test_pipeline".
    pipeline_logger = logging.getLogger("test_pipeline") 
    try:
        pipeline_logger.setLevel(args.log_level.upper())
        print(f"Уровень логирования для 'test_pipeline' установлен на: {args.log_level.upper()}")
    except ValueError:
        print(f"Некорректный уровень логирования: {args.log_level}. Используется INFO.")
        pipeline_logger.setLevel("INFO")


    print(f"Входной файл: {args.input_csv}")
    print(f"Выходной файл: {args.output_csv}")
    print(f"Директория артефактов: {args.artifacts_dir}")
    
    # Проверка существования входного файла
    if not os.path.exists(args.input_csv):
        print(f"Ошибка: Входной файл '{args.input_csv}' не найден.")
        print("Пожалуйста, убедитесь, что файл существует или укажите правильный путь.")
        exit(1) # Выход с кодом ошибки

    # Проверка существования директории с артефактами
    if not os.path.isdir(args.artifacts_dir):
        print(f"Ошибка: Директория с артефактами '{args.artifacts_dir}' не найдена.")
        print("Пожалуйста, убедитесь, что директория существует или укажите правильный путь.")
        exit(1)

    try:
        # Вызываем основную функцию из импортированного модуля
        test_pipeline.apply_predictions(
            input_csv_path=args.input_csv, 
            output_csv_path=args.output_csv, 
            artifacts_dir=args.artifacts_dir
        )
        print(f"Процесс предсказания завершен. Результаты сохранены в '{args.output_csv}'.")
    except FileNotFoundError as e:
        # Это исключение может быть поймано внутри apply_predictions,
        # но для надежности можно обработать и здесь.
        print(f"Ошибка: Файл не найден во время выполнения. {e}")
        pipeline_logger.error(f"Файл не найден: {e}", exc_info=True)
    except Exception as e:
        print(f"Произошла непредвиденная ошибка во время выполнения: {e}")
        # Логгер из test_pipeline уже должен был залогировать traceback,
        # но можно добавить дополнительное логирование здесь, если нужно.
        pipeline_logger.critical(f"Непредвиденная ошибка в test_data_predictor: {e}", exc_info=True)