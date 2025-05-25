# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Импорт функций из нашего модуля predictor
import predictor 
# Импорт опций для UI
try:
    import available_options as opts
    print("available_options.py успешно импортирован.")
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: available_options.py не найден. Пожалуйста, запустите generate_options.py.")
    # Создаем заглушки, чтобы приложение могло запуститься, но функциональность будет ограничена
    class opts:
        ALL_MPAA_RATINGS = ["PG-13", "R", "G"] # Минимальный набор для MPAA
        ALL_DIRECTORS, ALL_WRITERS, ALL_PRODUCERS, ALL_COMPOSERS, ALL_CINEMATOGRAPHERS, ALL_DISTRIBUTORS, ALL_ACTORS, ALL_GENRES = ([] for _ in range(8))
    # В продакшене лучше завершать работу, если этот файл критичен:
    # exit(1) 

app = Flask(__name__)

# Попытка инициализировать артефакты при старте Flask
try:
    print("Инициализация артефактов при старте Flask...")
    predictor.get_artifacts() 
    print("Артефакты успешно инициализированы (или уже были в кэше).")
except RuntimeError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ FLASK: Не удалось загрузить артефакты. {e}")
    # Приложение запустится, но /predict будет возвращать ошибку, пока артефакты не загрузятся.

@app.route('/')
def home():
    # Передаем только статические опции в шаблон, AJAX-опции будут загружаться динамически
    options_for_template = {
        "ALL_MPAA_RATINGS": opts.ALL_MPAA_RATINGS,
        # Если нужно передать какое-то начальное значение для AJAX-поля,
        # его можно добавить сюда и обработать в HTML <option selected>
        # "selected_director_default": "Some Default Director" 
    }
    return render_template('index.html', options=options_for_template)

@app.route('/get_options', methods=['GET'])
def get_options_for_select():
    search_term = request.args.get('q', default='', type=str).lower()
    field_name_from_request = request.args.get('field', default='', type=str)
    page = request.args.get('page', default=1, type=int)
    per_page = 30  # Количество элементов на страницу для Select2

    options_list_source = []
    # Сопоставление field_name из запроса с атрибутами в модуле opts
    if field_name_from_request == 'director': options_list_source = opts.ALL_DIRECTORS
    elif field_name_from_request == 'writer': options_list_source = opts.ALL_WRITERS
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
        filtered_options = options_list_source # Если нет поиска, возвращаем начало списка

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
            return jsonify({"error": "Пустой запрос или неверный Content-Type."}), 400

        input_data_for_model = {}
        # Определяем поля, которые ожидает модель (из column_info или явно)
        # Важно, чтобы сюда попали все поля, нужные для preprocess_input_data
        # включая те, на основе которых создаются признаки *_experience
        
        # Список полей, которые приходят с формы
        form_fields = [
            "movie_title", "movie_year", "budget", "run_time", "mpaa",
            "director", "writer", "producer", "composer", "cinematographer", "distributor",
            "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4",
            "genre_1", "genre_2", "genre_3", "genre_4"
        ]

        for field in form_fields:
            value = data_from_form.get(field)
            if isinstance(value, str) and value.strip() == "":
                input_data_for_model[field] = None 
            elif value == "null" or value is None:
                input_data_for_model[field] = None
            else:
                input_data_for_model[field] = value
        
        for num_field in ["budget", "movie_year"]:
            val = input_data_for_model.get(num_field)
            if val is not None:
                try: input_data_for_model[num_field] = float(val)
                except (ValueError, TypeError): input_data_for_model[num_field] = np.nan
            else: input_data_for_model[num_field] = np.nan

        predicted_gross = predictor.make_prediction(input_data_for_model)
        return jsonify({'prediction': predicted_gross})

    except ValueError as ve:
        print(f"Ошибка ValueError в /predict: {ve}")
        return jsonify({'error': f"Ошибка входных данных: {str(ve)}"}), 400
    except RuntimeError as re:
        print(f"Ошибка RuntimeError в /predict: {re}")
        return jsonify({'error': f"Ошибка сервера: {str(re)}"}), 500
    except Exception as e:
        print(f"Общая ошибка в /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Произошла непредвиденная ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) # Изменил порт на 5001 для примера