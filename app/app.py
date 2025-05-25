# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd # Нужен для pd.NA, pd.isna
import numpy as np  # Нужен для np.nan

# Импорт функций из нашего нового модуля
import predictor 
# Импорт опций для UI
import available_options as opts


app = Flask(__name__)

# Попытка загрузить артефакты при старте, чтобы проверить их доступность
# Фактическая загрузка и кэширование произойдет при первом вызове predictor.get_artifacts()
try:
    print("Попытка инициализации артефактов при старте Flask...")
    predictor.get_artifacts() 
    print("Артефакты успешно инициализированы (или уже были в кэше).")
except RuntimeError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ FLASK: Не удалось загрузить артефакты. {e}")
    # Приложение может работать некорректно или упасть при первом запросе /predict
    # В продакшене здесь можно было бы остановить запуск сервера.

@app.route('/')
def home():
    options_for_template = {
        "ALL_MPAA_RATINGS": opts.ALL_MPAA_RATINGS,
        "ALL_DIRECTORS": opts.ALL_DIRECTORS,
        "ALL_WRITERS": opts.ALL_WRITERS,
        "ALL_PRODUCERS": opts.ALL_PRODUCERS,
        "ALL_COMPOSERS": opts.ALL_COMPOSERS,
        "ALL_CINEMATOGRAPHERS": opts.ALL_CINEMATOGRAPHERS,
        "ALL_DISTRIBUTORS": opts.ALL_DISTRIBUTORS,
        "ALL_ACTORS": opts.ALL_ACTORS,
        "ALL_GENRES": opts.ALL_GENRES
    }
    return render_template('index.html', options=options_for_template)

@app.route('/predict', methods=['POST'])
def handle_predict_request():
    try:
        data_from_form = request.get_json()
        if not data_from_form:
            return jsonify({"error": "Пустой запрос или неверный Content-Type."}), 400

        # Подготовка входных данных для функции make_prediction
        # Преобразование пустых строк от формы в None, чтобы Pandas мог их корректно обработать как NaN
        input_data_for_model = {}
        # Список полей, которые мы ожидаем от формы и которые нужны для модели
        # Этот список должен соответствовать тем полям, которые используются в preprocess_input_data
        # (включая те, что нужны для создания признаков типа *_experience)
        expected_raw_fields = [
            "movie_title", "movie_year", "budget", "run_time", "mpaa",
            "director", "writer", "producer", "composer", "cinematographer", "distributor",
            "main_actor_1", "main_actor_2", "main_actor_3", "main_actor_4",
            "genre_1", "genre_2", "genre_3", "genre_4"
        ]
        
        for field in expected_raw_fields:
            value = data_from_form.get(field)
            # Преобразуем пустые строки в None. JS уже должен был это сделать, но для надежности.
            if isinstance(value, str) and value.strip() == "":
                input_data_for_model[field] = None
            elif value == "null" or value is None: # Если JS отправил строку "null" или явно null
                input_data_for_model[field] = None
            else:
                input_data_for_model[field] = value
        
        # Явное преобразование числовых полей, если они не None
        for num_field in ["budget", "movie_year"]:
            val = input_data_for_model.get(num_field)
            if val is not None:
                try:
                    input_data_for_model[num_field] = float(val)
                except (ValueError, TypeError):
                    input_data_for_model[num_field] = np.nan # Ошибка -> NaN
            else:
                input_data_for_model[num_field] = np.nan # None -> NaN

        # Вызов функции предсказания из модуля predictor
        predicted_gross = predictor.make_prediction(input_data_for_model)
        
        return jsonify({'prediction': predicted_gross})

    except ValueError as ve: # Ошибки типа данных или отсутствия ключей
        print(f"Ошибка ValueError в /predict: {ve}")
        return jsonify({'error': f"Ошибка входных данных: {str(ve)}"}), 400
    except RuntimeError as re: # Ошибки, связанные с артефактами
        print(f"Ошибка RuntimeError в /predict: {re}")
        return jsonify({'error': f"Ошибка сервера: {str(re)}"}), 500
    except Exception as e:
        print(f"Общая ошибка в /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Произошла непредвиденная ошибка: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)