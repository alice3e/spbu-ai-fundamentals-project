# generate_options.py (обновленный фрагмент)
import pandas as pd
import numpy as np
import os

# Путь к данным и файлу для сохранения опций
DATA_FILE_PATH = os.path.join("../data/", "all_data.csv")
OUTPUT_OPTIONS_FILE = "available_options.py"

def generate_and_save_options():
    print(f"Чтение данных из: {DATA_FILE_PATH}")
    try:
        data = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Ошибка: Файл {DATA_FILE_PATH} не найден.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла {DATA_FILE_PATH}: {e}")
        return

    print("Данные успешно загружены. Извлечение уникальных значений...")
    options = {}

    personnel_cols = ["director", "writer", "producer", "composer", "cinematographer", "distributor"]
    for col in personnel_cols:
        if col in data.columns:
            # Очищаем от пустых строк и NaN перед получением уникальных
            unique_values = data[col].astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()
            unique_values.sort()
            options[f"ALL_{col.upper()}S"] = unique_values
            print(f"  Найдено {len(unique_values)} уникальных значений для '{col}'.")
        else:
            options[f"ALL_{col.upper()}S"] = []

    actor_cols = [f"main_actor_{i}" for i in range(1, 5)]
    all_actors_set = set()
    for col in actor_cols:
        if col in data.columns:
            all_actors_set.update(data[col].astype(str).str.strip().replace('', np.nan).dropna().unique())
    all_actors_list = sorted(list(all_actors_set))
    options["ALL_ACTORS"] = all_actors_list
    print(f"  Найдено {len(all_actors_list)} уникальных актеров.")

    if "mpaa" in data.columns:
        unique_mpaa = data["mpaa"].astype(str).str.strip().replace('', np.nan).dropna().unique().tolist()
        unique_mpaa.sort()
        options["ALL_MPAA_RATINGS"] = unique_mpaa
        print(f"  Найдено {len(unique_mpaa)} уникальных MPAA рейтингов.")
    else:
        options["ALL_MPAA_RATINGS"] = []
    
    genre_cols = [f"genre_{i}" for i in range(1, 5)]
    all_genres_set = set()
    for col in genre_cols:
        if col in data.columns:
            all_genres_set.update(data[col].astype(str).str.strip().replace('', np.nan).dropna().unique())
    all_genres_list = sorted(list(all_genres_set))
    options["ALL_GENRES"] = all_genres_list
    print(f"  Найдено {len(all_genres_list)} уникальных жанров.")

    print(f"\nЗапись опций в файл: {OUTPUT_OPTIONS_FILE}")
    try:
        with open(OUTPUT_OPTIONS_FILE, "w", encoding="utf-8") as f:
            f.write("# Этот файл генерируется автоматически скриптом generate_options.py\n")
            f.write("# Содержит списки доступных опций для выбора пользователем.\n\n")
            for key, value_list in options.items():
                formatted_list_str = "[\n"
                for item in value_list:
                    item_str = str(item).replace('\\', '\\\\').replace("'", "\\'")
                    formatted_list_str += f"    '{item_str}',\n"
                formatted_list_str += "]"
                f.write(f"{key} = {formatted_list_str}\n\n")
        print("Файл с опциями успешно создан/обновлен.")
    except Exception as e:
        print(f"Ошибка при записи файла {OUTPUT_OPTIONS_FILE}: {e}")

if __name__ == "__main__":
    generate_and_save_options()