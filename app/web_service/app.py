# web_service/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd # Для pd.NA
import numpy as np  # Для np.nan
import requests # Для HTTP-запросов к сервису модели
import os

# Импорт опций для UI
try:
    import available_options as opts
    print("available_options.py успешно импортирован в web_service.")
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: available_options.py не найден в web_service.")
    class opts: 
        ALL_MPAA_RATINGS, ALL_DIRECTORS, ALL_WRITERS, ALL_PRODUCERS, ALL_COMPOSERS, ALL_CINEMATOGRAPHERS, ALL_DISTRIBUTORS, ALL_ACTORS, ALL_GENRES = ([] for _ in range(9))

app = Flask(__name__)

# Адрес сервиса модели (из Docker Compose или переменных окружения)
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://model_service:8000") # Имя сервиса из docker-compose

@app.route('/')
def home():
    options_for_template = {
        "ALL_MPAA_RATINGS": opts.ALL_MPAA_RATINGS,
        # AJAX-поля теперь не требуют передачи полных списков сюда
    }
    return render_template('index.html', options=options_for_template)

@app.route('/get_options', methods=['GET'])
def get_options_for_select():
    # Эта функция остается такой же, как в предыдущем ответе,
    # она использует локальный available_options.py
    search_term = request.args.get('q', default='', type=str).lower()
    field_name_from_request = request.args.get('field', default='', type=str)
    page = request.args.get('page', default=1, type=int)
    per_page = 30

    options_list_source = []
    if field_name_from_request == 'director': options_list_source = opts.ALL_DIRECTORS
    elif field_name_from_request == 'writer': options_list_source = opts.ALL_WRITERS
    # ... (остальные elif для других полей, как было раньше) ...
    elif field_name_from_request == 'producer': options_list_source = opts.ALL_PRODUCERS
    elif field_name_from_request == 'composer': options_list_source = opts.ALL_COMPOSERS
    elif field_name_from_request == 'cinematographer': options_list_source = opts.ALL_CINEMATOGRAPHERS
    elif field_name_from_request == 'distributor': options_list_source = opts.ALL_DISTRIBUTORS
    elif field_name_from_request.startswith('main_actor_'): options_list_source = opts.ALL_ACTORS
    elif field_name_from_request.startswith('genre_'): options_list_source = opts.ALL_GENRES

    if not options_list_source:
        return jsonify({"items": [], "pagination": {"more": False}})

    if search_term:
        filtered_options = [opt for opt in options_list_source if search_term in str(opt).lower()]
    else:
        filtered_options = options_list_source

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_options = filtered_options[start_index:end_index]
    select2_items = [{"id": opt, "text": opt} for opt in paginated_options]

    return jsonify({
        "items": select2_items,
        "pagination": {"more": end_index < len(filtered_options)}
    })

@app.route('/predict', methods=['POST'])
def handle_predict_request():
    try:
        data_from_form = request.get_json()
        if not data_from_form:
            return jsonify({"error": "Пустой запрос."}), 400

        # Данные, которые отправляются в сервис модели
        # Они должны соответствовать Pydantic модели в model_service/main.py
        # Преобразуем пустые строки и отсутствующие ключи в None
        payload_to_model_service = {}
        # Список полей, которые ожидает Pydantic модель в сервисе модели
        # (MovieDataInput в model_service/main.py)
        pydantic_model_fields = [
            "movie_title", "movie_year", "budget", "run_time", "mpaa",
            "director", "writer", "producer", "composer", "cinematographer", "distributor",
            "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4",
            "genre_1", "genre_2", "genre_3", "genre_4"
        ]

        for field in pydantic_model_fields:
            value = data_from_form.get(field)
            if isinstance(value, str) and value.strip() == "":
                payload_to_model_service[field] = None
            elif value == "null": # Если JS прислал строку "null"
                payload_to_model_service[field] = None
            else:
                payload_to_model_service[field] = value
        
        # Числовые поля - убедимся, что они числа или None
        for num_field in ["budget", "movie_year"]:
            val = payload_to_model_service.get(num_field)
            if val is not None:
                try:
                    payload_to_model_service[num_field] = float(val)
                except (ValueError, TypeError):
                    payload_to_model_service[num_field] = None # Ошибка -> None (Pydantic обработает)
            # else: # Если уже None, оставляем None

        predict_url = f"{MODEL_SERVICE_URL}/predict_movie_gross"
        
        print(f"Отправка запроса в сервис модели: {predict_url} с данными: {payload_to_model_service}")

        response_from_model = requests.post(predict_url, json=payload_to_model_service, timeout=10) # Таймаут 10 секунд
        response_from_model.raise_for_status() # Вызовет исключение для кодов 4xx/5xx

        prediction_result = response_from_model.json()
        
        return jsonify({'prediction': prediction_result.get('predicted_gross')})

    except requests.exceptions.RequestException as req_err:
        print(f"Ошибка соединения с сервисом модели: {req_err}")
        return jsonify({'error': f"Не удалось связаться с сервисом предсказаний: {str(req_err)}"}), 503
    except Exception as e:
        print(f"Общая ошибка в /predict (web_service): {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Произошла непредвиденная ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    # Запуск Flask приложения
    # Убедитесь, что generate_options.py был запущен и available_options.py существует
    # перед запуском этого приложения.
    # generate_options.py и all_data.csv должны быть в web_service, если generate_options.py запускается отсюда.
    # Но лучше генерировать available_options.py один раз и копировать.
    
    # Для локального запуска: python web_service/app.py
    # Docker будет использовать команду из Dockerfile
    app.run(debug=True, host='0.0.0.0', port=5019)