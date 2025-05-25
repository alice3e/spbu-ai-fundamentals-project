#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import random
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import category_encoders as ce
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os


# In[ ]:


# --- Настройки и глобальные переменные ---
RANDOM_SEED = 42
TEST_SET_SIZE = 0.3
MODEL_SAVE_DIR = "saved_model"

# Инициализация словарей для карт преобразований (будут заполняться в процессе)
experience_maps_to_save = {}
grouped_imputation_maps_to_save = {}

# Колонки, которые будут удалены из данных в самом начале
# Они не являются предикторами или являются прямыми компонентами/сильными коррелятами target
# 'worldwide' - это таргет, он будет отделен позже.
# 'domestic' и 'international' - прямые компоненты 'worldwide', их нужно удалить, чтобы избежать утечки.
INITIAL_COLUMNS_TO_DROP_CONFIG = ['movie_id', 'movie_title', 'link', 'domestic', 'international']


# In[ ]:


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# In[ ]:


# --- Загрузка и первичная обработка данных ---
print("Загрузка данных...")
try:
    data = pd.read_csv("all_data.csv")
    print(f"Данные загружены. Форма: {data.shape}")
except FileNotFoundError:
    print("Ошибка: Файл 'all_data.csv' не найден. Убедитесь, что он находится в той же директории.")
    exit()

print("\nПервые 5 строк исходных данных:")
print(data.head())

print("\nКолонки в исходных данных:")
print(data.columns.tolist())

# Проверка пропусков в целевой переменной
print(f"\nПропуски в 'worldwide' до удаления: {data['worldwide'].isnull().sum()}")
data = data.dropna(subset=['worldwide'])
print(f"Строки с NaN в 'worldwide' удалены. Новая форма: {data.shape}")


# In[ ]:


# --- Разделение на обучающую и тестовую выборки ---
# Тестовая выборка сохраняется для финальной оценки, но не используется для обучения трансформеров
print("\nРазделение на обучающую и тестовую выборки...")
train_data, test_data_for_final_eval = train_test_split(data, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED)
print(f"Обучающая выборка: {train_data.shape}")
print(f"Тестовая выборка (для финальной оценки): {test_data_for_final_eval.shape}")

# Сохраняем тестовый набор для последующего использования с test_pipeline.py
test_data_for_final_eval.to_csv("test.csv", index=False)
print("Тестовый набор (test.csv) сохранен.")


# In[ ]:


# --- Удаление начальных колонок из обучающей выборки ---
# Эти колонки не будут участвовать в обучении модели или препроцессоров
print(f"\nУдаление начальных колонок из обучающей выборки: {INITIAL_COLUMNS_TO_DROP_CONFIG}")
train_data = train_data.drop(columns=[col for col in INITIAL_COLUMNS_TO_DROP_CONFIG if col in train_data.columns], errors='ignore')
print(f"Форма обучающей выборки после удаления колонок: {train_data.shape}")
print("Колонки в train_data после начального удаления:", train_data.columns.tolist())


# In[ ]:


# --- Визуализация и EDA (на обучающей выборке) ---
# (Можно добавить или убрать блоки EDA по необходимости)
print("\nНачало EDA на обучающей выборке...")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Распределение целевой переменной 'worldwide'
plt.figure(figsize=(10, 5))
sns.histplot(train_data['worldwide'], kde=True, bins=50)
plt.title(f'Распределение worldwide (Обучающая выборка)\nSkewness: {train_data["worldwide"].skew():.2f}')
plt.xlabel('Сборы ($)')
plt.ylabel('Частота')
plt.tight_layout()
plt.show()

# Примеры других EDA (бюджет, год)
if 'budget' in train_data.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(train_data['budget'].dropna(), kde=True, bins=30)
    plt.title("Распределение 'budget'")
    plt.show()

if 'movie_year' in train_data.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='movie_year', y='worldwide', data=train_data.groupby('movie_year')['worldwide'].median().reset_index())
    plt.title("Медианные сборы 'worldwide' по годам")
    plt.show()
print("EDA завершен.")


# In[ ]:


# --- Feature Engineering ---

# 1. Создание признаков опыта и СОХРАНЕНИЕ КАРТ
print("\nШаг 1: Feature Engineering (Опыт)")
personnel_cols_for_experience = ["director", "writer", "producer", "composer", "cinematographer"]
for col in personnel_cols_for_experience:
    if col in train_data.columns:
        counts = train_data[col].value_counts()
        experience_maps_to_save[f"{col}_experience_map"] = counts.to_dict()
        train_data[f"{col}_experience"] = train_data[col].map(counts).fillna(0)
        print(f"  Создан признак {col}_experience и сохранена карта.")
    else:
        print(f"  Предупреждение: столбец {col} не найден для создания опыта.")

actor_experience_cols_generated = []
actor_cols_for_experience_prefix = "main_actor_" # Сохраняем для column_info
for i in range(1, 5):
    col_actor = f"{actor_cols_for_experience_prefix}{i}"
    if col_actor in train_data.columns:
        counts_actor = train_data[col_actor].value_counts()
        experience_maps_to_save[f"{col_actor}_experience_map"] = counts_actor.to_dict()
        experience_col_name = f"{col_actor}_experience"
        train_data[experience_col_name] = train_data[col_actor].map(counts_actor).fillna(0)
        actor_experience_cols_generated.append(experience_col_name)
        print(f"  Создан признак {experience_col_name} и сохранена карта.")
    else:
        print(f"  Предупреждение: столбец {col_actor} не найден для создания опыта.")

if actor_experience_cols_generated:
    train_data["cast_popularity"] = train_data[actor_experience_cols_generated].sum(axis=1)
    print("  Создан признак cast_popularity.")
else:
    train_data["cast_popularity"] = 0
    print("  Признак cast_popularity установлен в 0 (колонки опыта актеров не созданы).")

# 2. Преобразование run_time
print("\nШаг 2: Преобразование 'run_time'")
def convert_runtime_to_minutes_func(value): # Переименовал, чтобы не конфликтовать с внешней функцией
    if pd.isna(value) or value == "": return None
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

if 'run_time' in train_data.columns:
    train_data['run_time'] = train_data['run_time'].apply(convert_runtime_to_minutes_func)
    print("  'run_time' преобразован в минуты.")
else:
    print("  Предупреждение: столбец 'run_time' не найден.")


# In[ ]:


# --- Импутация пропусков ---

# 3. Numerical Imputation
print("\nШаг 3: Numerical Imputation (медианой)")
# Исключаем целевую переменную и уже созданные признаки опыта (если они попали в числовые)
# domestic и international уже должны быть удалены
numerical_cols_for_imputation = [
    col for col in train_data.select_dtypes(include=[np.number]).columns
    if col not in ['worldwide'] and not col.endswith("_experience") and col != "cast_popularity"
]
print(f"  Числовые колонки для импутации: {numerical_cols_for_imputation}")

if numerical_cols_for_imputation:
    num_imputer = SimpleImputer(strategy='median')
    train_data[numerical_cols_for_imputation] = \
        num_imputer.fit_transform(train_data[numerical_cols_for_imputation])
    print(f"  Numerical Imputer обучен на колонках: {num_imputer.feature_names_in_.tolist()}")
    numerical_imputer_features_in_list = num_imputer.feature_names_in_.tolist()
else:
    num_imputer = None
    numerical_imputer_features_in_list = []
    print("  ПРЕДУПРЕЖДЕНИЕ: Нет числовых колонок для обучения Numerical Imputer.")

# 4. Grouped Imputation (модой по 'director')
print("\nШаг 4: Grouped Imputation (модой по 'director')")
cols_for_grouped_imputation_source_director_list = ['cinematographer', 'composer', 'producer', 'writer']
if 'director' in train_data.columns:
    for col_to_impute in cols_for_grouped_imputation_source_director_list:
        if col_to_impute in train_data.columns:
            director_to_mode_map = train_data.groupby('director')[col_to_impute].apply(
                lambda x: x.mode().iloc[0] if not x.mode().empty and pd.notna(x.mode().iloc[0]) else np.nan
            )
            # Заменяем NaN ключи (если есть) на строку "NaN_director_placeholder" или удаляем
            # Для простоты, предполагаем, что имена режиссеров не NaN. Если могут быть, нужна обработка.
            grouped_imputation_maps_to_save[f"{col_to_impute}_director_mode_map"] = director_to_mode_map.to_dict()
            
            mapped_modes_for_imputation = train_data['director'].map(director_to_mode_map)
            train_data[col_to_impute].fillna(mapped_modes_for_imputation, inplace=True)
            print(f"    Пропуски в '{col_to_impute}' заполнены модами по 'director'. Карта сохранена.")
        else:
            print(f"  Предупреждение: столбец '{col_to_impute}' не найден для групповой импутации.")
else:
    print("  Предупреждение: столбец 'director' не найден, групповая импутация не будет выполнена.")

# 5. Заполнение 'Unknown' для специфических категориальных колонок
print("\nШаг 5: Заполнение 'Unknown' для специфических колонок")
cols_to_fill_unknown_specific_list = ['genre_2', 'genre_3', 'genre_4', 'main_actor_4']
for col in cols_to_fill_unknown_specific_list:
    if col in train_data.columns:
        train_data[col] = train_data[col].apply(lambda x: 'Unknown' if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x)
        print(f"    '{col}' обработан для 'Unknown'.")
    else:
        print(f"  Предупреждение: столбец '{col}' не найден для заполнения 'Unknown'.")


# 6. Categorical Imputation (общее, модой)
print("\nШаг 6: Categorical Imputation (общее, модой)")
# Выбираем все оставшиеся object колонки, которые могут иметь пропуски
# Это включает и те, что были частично обработаны в Шаге 4 и 5, если там остались NaN
cat_cols_for_general_imputation = train_data.select_dtypes(include=['object']).columns.tolist()
print(f"  Категориальные колонки для общей импутации: {cat_cols_for_general_imputation}")

if cat_cols_for_general_imputation:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_data[cat_cols_for_general_imputation] = \
        cat_imputer.fit_transform(train_data[cat_cols_for_general_imputation])
    print(f"  Categorical Imputer обучен на колонках: {cat_imputer.feature_names_in_.tolist()}")
    categorical_imputer_features_in_list = cat_imputer.feature_names_in_.tolist()
else:
    cat_imputer = None
    categorical_imputer_features_in_list = []
    print("  ПРЕДУПРЕЖДЕНИЕ: Нет категориальных колонок для обучения Categorical Imputer.")


# In[ ]:


# --- Кодирование категориальных признаков ---
print("\nПроверка уникальных значений в категориальных колонках перед кодированием:")
cat_cols_final_check = train_data.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols_final_check:
    cat_stats = pd.DataFrame({
        'Column': cat_cols_final_check,
        'Unique_Count': [train_data[col].nunique() for col in cat_cols_final_check],
    }).sort_values(by='Unique_Count', ascending=False)
    print(cat_stats)
else:
    print("  Нет категориальных колонок для кодирования.")


# Определяем колонки для OHE и Target Encoding
# Важно: эти списки должны содержать только те колонки, которые существуют в train_data на данном этапе
# и которые действительно являются категориальными.
ohe_input_columns_list = [
    col for col in ['genre_1', 'genre_2', 'genre_3', 'genre_4', 'mpaa']
    if col in train_data.columns and train_data[col].dtype == 'object'
]
target_encoding_columns_list = [
    col for col in ['main_actor_1', 'main_actor_2', 'main_actor_3', 'main_actor_4',
                    'director', 'writer', 'producer', 'composer', 'cinematographer', 'distributor']
    if col in train_data.columns and train_data[col].dtype == 'object'
]
print(f"\nКолонки для One-Hot Encoding: {ohe_input_columns_list}")
print(f"Колонки для Target Encoding: {target_encoding_columns_list}")


# 7. One-Hot Encoding
print("\nШаг 7: One-Hot Encoding")
if ohe_input_columns_list:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Перед fit/transform убедимся, что в колонках нет NaN (должны были быть устранены импутацией)
    # и что они строкового типа
    for col in ohe_input_columns_list:
        if train_data[col].isnull().any():
             print(f"  ВНИМАНИЕ: NaN найдены в '{col}' перед OHE. Заполняем модой.")
             train_data[col].fillna(train_data[col].mode()[0], inplace=True)
        train_data[col] = train_data[col].astype(str)

    ohe.fit(train_data[ohe_input_columns_list])
    train_encoded_ohe = ohe.transform(train_data[ohe_input_columns_list])
    ohe_feature_names = ohe.get_feature_names_out(ohe_input_columns_list)
    train_encoded_ohe_df = pd.DataFrame(train_encoded_ohe, columns=ohe_feature_names, index=train_data.index)
    
    train_data = train_data.drop(columns=ohe_input_columns_list)
    train_data = pd.concat([train_data, train_encoded_ohe_df], axis=1)
    print(f"  OneHotEncoder применен. Добавлено {len(ohe_feature_names)} колонок.")
else:
    ohe = None # Если нет колонок для OHE
    print("  Нет колонок для One-Hot Encoding.")

# 8. Target Encoding
print("\nШаг 8: Target Encoding")
if target_encoding_columns_list:
    target_encoder = ce.TargetEncoder(cols=target_encoding_columns_list, handle_unknown='value', handle_missing='value')
    # 'value' для handle_unknown/handle_missing означает, что для неизвестных/пропущенных значений
    # будет использовано среднее значение таргета по всем обучающим данным.
    # Убедимся, что нет NaN перед TE и тип строковый
    for col in target_encoding_columns_list:
        if train_data[col].isnull().any():
             print(f"  ВНИМАНИЕ: NaN найдены в '{col}' перед Target Encoding. Заполняем модой.")
             train_data[col].fillna(train_data[col].mode()[0], inplace=True)
        train_data[col] = train_data[col].astype(str)

    train_data[target_encoding_columns_list] = target_encoder.fit_transform(train_data[target_encoding_columns_list], train_data['worldwide'])
    print("  TargetEncoder применен.")
else:
    target_encoder = None # Если нет колонок для TE
    print("  Нет колонок для Target Encoding.")

print(f"\nФорма train_data после кодирования: {train_data.shape}")
print("Пример колонок после кодирования (первые 15):", train_data.columns.tolist()[:15])


# In[ ]:


# --- Подготовка данных для модели (X, y) ---
print("\nПодготовка X и y для модели...")
if 'worldwide' not in train_data.columns:
    raise ValueError("'worldwide' отсутствует в train_data перед разделением на X и y.")

X_train_final = train_data.drop(columns=['worldwide'])
y_train_final = train_data['worldwide']

print(f"  Форма X_train_final: {X_train_final.shape}")
print(f"  Форма y_train_final: {y_train_final.shape}")
final_model_features_list_before_scaling = X_train_final.columns.tolist()


# In[ ]:


# --- Масштабирование признаков ---
print("\nШаг 9: Масштабирование признаков (StandardScaler)")
# Убедимся, что нет NaN в X_train_final перед масштабированием (на всякий случай)
if X_train_final.isnull().any().any():
    print("  ВНИМАНИЕ: Обнаружены NaN в X_train_final перед масштабированием. Заполняем медианой колонки.")
    # Более надежно было бы использовать num_imputer, если бы он обучался на всех числовых фичах X
    # Но для простоты, если NaN остались (не должны), заполним медианой по колонке
    for col in X_train_final.columns[X_train_final.isnull().any()]:
        if pd.api.types.is_numeric_dtype(X_train_final[col]):
            X_train_final[col].fillna(X_train_final[col].median(), inplace=True)
        else: # Если вдруг нечисловая колонка с NaN осталась (очень маловероятно здесь)
            X_train_final[col].fillna(X_train_final[col].mode()[0], inplace=True)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
print("  Признаки X_train_final масштабированы.")


# In[ ]:

# In[ ]:

# --- Обучение модели с RandomizedSearchCV (на всех X_train_scaled) ---
print("\nОбучение модели XGBRegressor с RandomizedSearchCV (на всех X_train_scaled)...")

# Предыдущий train_test_split для X_model_train, X_model_val УБРАН.
# RandomizedSearchCV будет использовать кросс-валидацию на X_train_scaled.

param_dist = {
    'n_estimators': [200, 400, 600, 800, 1000], # Расширенный диапазон
    'max_depth':    [4, 5, 6, 7, 8, 10],      # Расширенный диапазон
    'learning_rate':[0.01, 0.02, 0.03, 0.05, 0.1], # Более гранулированный learning rate
    'subsample':    [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],   # Более гранулированный colsample
    'gamma': [0, 0.05, 0.1, 0.2, 0.3],        # Параметр для контроля сложности дерева
    'reg_alpha':    [0, 0.01, 0.05, 0.1, 0.5, 1], # Добавлены промежуточные значения для регуляризации
    'reg_lambda':   [0.5, 1, 1.5, 2, 3, 5]       # Добавлены промежуточные значения для регуляризации
}

base_model_xgb = XGBRegressor(
    random_state=RANDOM_SEED,
    tree_method='hist', 
    n_jobs=-1 
)

random_search_xgb = RandomizedSearchCV(
    estimator=base_model_xgb,
    param_distributions=param_dist,
    n_iter=75,  # Увеличено количество итераций для более тщательного поиска
    scoring='r2', # Используем R2 напрямую для оптимизации
    cv=5,       # 5-фолдовая кросс-валидация (более надежно, чем 3)
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=1    # n_jobs для RandomizedSearchCV (не для XGB), 1 для воспроизводимости
)

# Обучаем RandomizedSearchCV на ВСЕХ доступных тренировочных масштабированных данных
random_search_xgb.fit(X_train_scaled, y_train_final)

# Получаем лучшие параметры
best_params_from_search = random_search_xgb.best_params_
print("\nЛучшие параметры для XGBRegressor из RandomizedSearchCV:", best_params_from_search)

# Получаем лучшую оценку R2, полученную на кросс-валидации
best_cv_r2_score = random_search_xgb.best_score_
print(f"Лучший R2 на кросс-валидации (средний по фолдам): {best_cv_r2_score:.4f}")

# best_estimator_ из RandomizedSearchCV уже обучен на всех данных (если refit=True, что по умолчанию)
# с использованием лучших параметров. Однако, для применения Early Stopping, мы обычно
# создаем новую модель с этими параметрами и обучаем ее отдельно с eval_set.
# Поэтому, здесь мы просто сохраняем best_params_from_search для следующего шага.
# Экземпляр random_search_xgb.best_estimator_ можно было бы использовать, если бы не Early Stopping.

# Блок оценки на X_model_val теперь не нужен, так как best_cv_r2_score - более надежная метрика.
# y_val_pred = best_xgb_model.predict(X_model_val)
# print("\nМетрики на валидационном наборе (внутри RandomizedSearchCV):") # Это было бы некорректно называть "внутри RandomizedSearchCV"
# ... (старый блок оценки удален)

# In[ ]: # Следующая ячейка

# --- Обучение финальной модели с лучшими параметрами и Early Stopping ---
print("\nОбучение финальной модели с лучшими параметрами и Early Stopping...")

# Разделяем X_train_scaled для создания eval_set для early stopping
X_final_train_es, X_final_eval_es, y_final_train_es, y_final_eval_es = train_test_split(
    X_train_scaled, y_train_final, test_size=0.2, random_state=RANDOM_SEED + 1 # Другой seed для воспроизводимости этого шага
)

# Создаем модель, передавая early_stopping_rounds в конструктор
# и лучшие параметры из RandomizedSearchCV
final_best_xgb_model = XGBRegressor(
    **best_params_from_search,     # <--- Используем параметры, найденные выше
    early_stopping_rounds=50,      
    random_state=RANDOM_SEED,
    tree_method='hist',
    n_jobs=-1
)

# При вызове fit передаем eval_set
final_best_xgb_model.fit(
    X_final_train_es, y_final_train_es,
    eval_set=[(X_final_eval_es, y_final_eval_es)],
    verbose=False 
)

print(f"Финальная модель обучена. Количество деревьев: {final_best_xgb_model.get_booster().num_boosted_rounds()}")

# Эта final_best_xgb_model теперь будет вашей моделью для сохранения и использования
# Переименуем ее в best_xgb_model для совместимости с остальным кодом
best_xgb_model = final_best_xgb_model 

# In[ ]:


# --- Сохранение артефактов ---
print(f"\n--- СОХРАНЕНИЕ АРТЕФАКТОВ в директорию: {MODEL_SAVE_DIR} ---")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# 1. Модель
# Сохраняем как XGBRegressor (scikit-learn wrapper)
model_filename_joblib = os.path.join(MODEL_SAVE_DIR, "movie_box_office_model.joblib")
joblib.dump(best_xgb_model, model_filename_joblib)
print(f"Модель (XGBRegressor) сохранена: {model_filename_joblib}")

# Сохраняем "сырой" XGBoost Booster в JSON формате (если нужен для predictor.py)
if hasattr(best_xgb_model, 'get_booster'):
    model_filename_json = os.path.join(MODEL_SAVE_DIR, "movie_box_office_model.json")
    booster_to_save = best_xgb_model.get_booster()
    booster_to_save.save_model(model_filename_json)
    print(f"Модель (XGBoost Booster) сохранена в JSON: {model_filename_json}")
else:
    print("Не удалось получить XGBoost Booster из best_xgb_model для сохранения в JSON.")


# 2. Трансформеры
if num_imputer:
    joblib.dump(num_imputer, os.path.join(MODEL_SAVE_DIR, "numerical_imputer.joblib"))
    print("Numerical Imputer сохранен.")
if cat_imputer:
    joblib.dump(cat_imputer, os.path.join(MODEL_SAVE_DIR, "categorical_imputer.joblib"))
    print("Categorical Imputer сохранен.")
if ohe:
    joblib.dump(ohe, os.path.join(MODEL_SAVE_DIR, "onehot_encoder.joblib"))
    print("OneHotEncoder сохранен.")
if target_encoder:
    joblib.dump(target_encoder, os.path.join(MODEL_SAVE_DIR, "target_encoder.joblib"))
    print("Target Encoder сохранен.")
if scaler: # scaler должен быть всегда, если X_train_final не пустой
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.joblib"))
    print("Scaler сохранен.")

# 3. Информация о колонках (column_info.json)
# Убедимся, что списки колонок для JSON корректны
# final_model_features_list_before_scaling - это X.columns ПЕРЕД масштабированием,
# но ПОСЛЕ всех преобразований (OHE, TE и т.д.). Это то, что ожидает scaler.
column_info_data = {
    "features": final_model_features_list_before_scaling,
    "target": "worldwide",
    "initial_columns_to_drop": INITIAL_COLUMNS_TO_DROP_CONFIG,
    "ohe_input_columns": ohe_input_columns_list if ohe else [],
    "target_encoding_columns": target_encoding_columns_list if target_encoder else [],
    "numerical_imputer_features_in": numerical_imputer_features_in_list if num_imputer else [],
    "categorical_imputer_features_in": categorical_imputer_features_in_list if cat_imputer else [],
    "experience_maps": experience_maps_to_save,
    "grouped_imputation_maps": grouped_imputation_maps_to_save,
    "personnel_cols_for_experience": personnel_cols_for_experience,
    "actor_cols_for_experience_prefix": actor_cols_for_experience_prefix,
    "cols_for_grouped_imputation_source_director": cols_for_grouped_imputation_source_director_list,
    "cols_to_fill_unknown_specific": cols_to_fill_unknown_specific_list
}

# Очистка np.nan для JSON сериализации в картах
def clean_map_for_json(data_map):
    if not isinstance(data_map, dict): return {}
    cleaned = {}
    for map_key, inner_dict in data_map.items():
        if isinstance(inner_dict, dict):
            cleaned_inner = {}
            for k, v in inner_dict.items():
                # Ключи должны быть строками. Значения None, если были NaN.
                cleaned_inner[str(k)] = None if pd.isna(v) else v
            cleaned[map_key] = cleaned_inner
        else:
            cleaned[map_key] = None if pd.isna(inner_dict) else inner_dict
    return cleaned

if column_info_data["experience_maps"]:
    column_info_data["experience_maps"] = clean_map_for_json(column_info_data["experience_maps"])
if column_info_data["grouped_imputation_maps"]:
    column_info_data["grouped_imputation_maps"] = clean_map_for_json(column_info_data["grouped_imputation_maps"])

column_info_filename = os.path.join(MODEL_SAVE_DIR, "column_info.json")
with open(column_info_filename, "w", encoding='utf-8') as f:
    json.dump(column_info_data, f, indent=2, ensure_ascii=False, default=lambda x: None if pd.isna(x) else x)
print(f"Информация о колонках сохранена: {column_info_filename}")

print("\n>>> СОХРАНЕНИЕ АРТЕФАКТОВ ЗАВЕРШЕНО <<<")


# In[ ]:

import xgboost as xgb
# --- Финальная проверка: Загрузить артефакты и предсказать на сохраненном test.csv ---
# Этот блок эмулирует работу test_pipeline.py для быстрой проверки
print("\n--- Финальная проверка с test_pipeline.py логикой на test.csv ---")

# Скопируем сюда основные функции из test_pipeline.py (упрощенно) для проверки
def log_dataframe_info_check(df, name="DataFrame", max_rows=3):
    print(f"Инфо о '{name}': Форма: {df.shape}")
    if not df.empty: print(f"  Первые {min(max_rows, df.shape[0])} строк:\n{df.head(max_rows).to_string()}")

def load_all_artifacts_check(artifacts_dir="saved_model"):
    # Упрощенная загрузка, предполагая, что все joblib/json файлы существуют
    artifacts = {}
    artifacts["column_info"] = json.load(open(os.path.join(artifacts_dir, "column_info.json"), 'r', encoding='utf-8'))
    if os.path.exists(os.path.join(artifacts_dir, "movie_box_office_model.json")):
        model_obj = xgb.Booster()
        model_obj.load_model(os.path.join(artifacts_dir, "movie_box_office_model.json"))
        artifacts["model"] = model_obj
    elif os.path.exists(os.path.join(artifacts_dir, "movie_box_office_model.joblib")):
         artifacts["model"] = joblib.load(os.path.join(artifacts_dir, "movie_box_office_model.joblib"))
    else:
        raise FileNotFoundError("Модель не найдена в saved_model")

    for f_name_part in ["numerical_imputer", "categorical_imputer", "onehot_encoder", "target_encoder", "scaler"]:
        path = os.path.join(artifacts_dir, f"{f_name_part}.joblib")
        if os.path.exists(path): artifacts[f_name_part] = joblib.load(path)
        else: print(f"Предупреждение: Артефакт {f_name_part}.joblib не найден для проверки.")
    return artifacts

def preprocess_dataframe_check(df_input: pd.DataFrame, artifacts: dict):
    # Упрощенная и адаптированная версия preprocess_dataframe из test_pipeline.py
    # ВАЖНО: Эта версия должна точно соответствовать логике test_pipeline.py
    df = df_input.copy()
    column_info = artifacts['column_info']

    # Шаг 0: Удаление начальных колонок
    initial_cols_to_drop = column_info.get("initial_columns_to_drop", [])
    if initial_cols_to_drop:
        cols_present_to_drop = [col for col in initial_cols_to_drop if col in df.columns]
        if cols_present_to_drop: df = df.drop(columns=cols_present_to_drop)

    # Шаг 1: Feature Engineering (опыт)
    exp_maps = column_info.get("experience_maps", {})
    personnel_cols = column_info.get("personnel_cols_for_experience", [])
    for col in personnel_cols:
        map_key = f"{col}_experience_map"
        if col in df.columns and map_key in exp_maps:
            df[f"{col}_experience"] = df[col].astype(str).map(exp_maps[map_key]).fillna(0)
        elif col in df.columns: df[f"{col}_experience"] = 0
    
    actor_exp_cols_gen = []
    actor_prefix = column_info.get("actor_cols_for_experience_prefix", "main_actor_")
    for i in range(1, 5):
        col_actor = f"{actor_prefix}{i}"
        map_key_actor = f"{col_actor}_experience_map"
        if col_actor in df.columns and map_key_actor in exp_maps:
            exp_col_name = f"{col_actor}_experience"
            df[exp_col_name] = df[col_actor].astype(str).map(exp_maps[map_key_actor]).fillna(0)
            actor_exp_cols_gen.append(exp_col_name)
        elif col_actor in df.columns:
            df[f"{col_actor}_experience"] = 0
            actor_exp_cols_gen.append(f"{col_actor}_experience")
    if actor_exp_cols_gen: df["cast_popularity"] = df[actor_exp_cols_gen].sum(axis=1)
    else: df["cast_popularity"] = 0

    # Шаг 2: Преобразование run_time
    if 'run_time' in df.columns:
        df['run_time'] = df['run_time'].apply(convert_runtime_to_minutes_func) # Используем ту же функцию

    # Шаг 3: Numerical Imputation
    num_imputer = artifacts.get('numerical_imputer')
    num_features_expected = column_info.get("numerical_imputer_features_in", [])
    if num_imputer and num_features_expected:
        for col in num_features_expected:
            if col not in df.columns: df[col] = np.nan
        df_subset_num = df[num_features_expected].copy()
        imputed_values = num_imputer.transform(df_subset_num)
        df[num_features_expected] = pd.DataFrame(imputed_values, columns=num_features_expected, index=df.index)

    # Шаг 4: Grouped Imputation
    group_maps = column_info.get("grouped_imputation_maps", {}) 
    group_impute_cols = column_info.get("cols_for_grouped_imputation_source_director", []) 
    if 'director' in df.columns and group_impute_cols:
        for col_to_impute in group_impute_cols:
            if col_to_impute not in df.columns: df[col_to_impute] = np.nan
            map_key = f"{col_to_impute}_director_mode_map"
            current_col_map = group_maps.get(map_key, {})
            mapped_modes = df['director'].astype(str).map(current_col_map) 
            if col_to_impute in df.columns: df[col_to_impute].fillna(mapped_modes, inplace=True)

    # Шаг 5: Fill 'Unknown'
    fill_unknown_cols = column_info.get("cols_to_fill_unknown_specific", []) 
    if fill_unknown_cols:
        for col in fill_unknown_cols:
            if col not in df.columns: df[col] = np.nan
            df[col] = df[col].apply(lambda x: 'Unknown' if pd.isna(x) or (isinstance(x, str) and x.strip() == '') else x)

    # Шаг 6: Categorical Imputation
    cat_imputer = artifacts.get('categorical_imputer')
    cat_features_expected = column_info.get("categorical_imputer_features_in", [])
    if cat_imputer and cat_features_expected:
        for col in cat_features_expected:
            if col not in df.columns: df[col] = np.nan
        df_subset_cat = df[cat_features_expected].copy()
        imputed_cat_values = cat_imputer.transform(df_subset_cat)
        df[cat_features_expected] = pd.DataFrame(imputed_cat_values, columns=cat_features_expected, index=df.index)
    
    # Шаг 7: One-Hot Encoding
    ohe_encoder = artifacts.get('onehot_encoder')
    ohe_input_cols = column_info.get("ohe_input_columns", [])
    if ohe_encoder and ohe_input_cols:
        for col in ohe_input_cols:
            if col not in df.columns: df[col] = "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)
        ohe_transformed = ohe_encoder.transform(df[ohe_input_cols])
        ohe_names = ohe_encoder.get_feature_names_out(ohe_input_cols)
        ohe_df = pd.DataFrame(ohe_transformed.toarray() if hasattr(ohe_transformed, "toarray") else ohe_transformed,
                              columns=ohe_names, index=df.index)
        df = df.drop(columns=ohe_input_cols, errors='ignore')
        df = pd.concat([df, ohe_df], axis=1)

    # Шаг 8: Target Encoding
    te_encoder = artifacts.get('target_encoder')
    te_input_cols = column_info.get("target_encoding_columns", [])
    if te_encoder and te_input_cols:
        cols_for_te_transform = []
        for col in te_input_cols:
            if col not in df.columns: df[col] = "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)
            cols_for_te_transform.append(col)
        if cols_for_te_transform: # Убедимся, что есть что трансформировать
            transformed_te_vals = te_encoder.transform(df[cols_for_te_transform])
            df[cols_for_te_transform] = transformed_te_vals

    # Шаг 9: Reindex
    final_features = column_info.get("features")
    df_processed = pd.DataFrame(0.0, index=df.index, columns=final_features)
    common_cols = df_processed.columns.intersection(df.columns)
    if not common_cols.empty: df_processed[common_cols] = df[common_cols]
    df_processed.fillna(0, inplace=True) # Важно для scaler

    # Шаг 10: Scaling
    scaler_obj = artifacts.get('scaler')
    if not scaler_obj: raise ValueError("Scaler не найден в артефактах для проверки")
    X_scaled_np = scaler_obj.transform(df_processed)
    return X_scaled_np, final_features

try:
    artifacts_check = load_all_artifacts_check()
    df_test_raw_check = pd.read_csv("test.csv")
    log_dataframe_info_check(df_test_raw_check, "df_test_raw_check (исходный для проверки)")
    
    # Сохраняем 'worldwide' из тестового файла для сравнения, если он там есть
    y_true_test_check = None
    if 'worldwide' in df_test_raw_check.columns:
        y_true_test_check = df_test_raw_check['worldwide'].copy()

    processed_data_np_check, feature_names_check = preprocess_dataframe_check(df_test_raw_check.copy(), artifacts_check)
    
    log_dataframe_info_check(pd.DataFrame(processed_data_np_check, columns=feature_names_check), "processed_data_np_check (для DMatrix)")

    model_check = artifacts_check.get('model')
    if not model_check: raise ValueError("Модель не найдена в артефактах для проверки")

    dmatrix_processed_check = xgb.DMatrix(processed_data_np_check, feature_names=feature_names_check)
    predictions_raw_check = model_check.predict(dmatrix_processed_check)
    
    df_test_raw_check['predicted_worldwide_check'] = np.round(predictions_raw_check).astype(float)
    print("\nПредсказания на test.csv (проверочные, первые 5 строк):")
    cols_to_show_check = ['movie_title', 'predicted_worldwide_check']
    if y_true_test_check is not None:
        cols_to_show_check.insert(1, 'worldwide_actual')
        df_test_raw_check['worldwide_actual'] = y_true_test_check
    
    print(df_test_raw_check[cols_to_show_check].head())

    if y_true_test_check is not None:
        print("\nМетрики на test.csv (проверочные):")
        r2_check = r2_score(y_true_test_check, df_test_raw_check['predicted_worldwide_check'])
        mse_check = mean_squared_error(y_true_test_check, df_test_raw_check['predicted_worldwide_check'])
        mae_check = mean_absolute_error(y_true_test_check, df_test_raw_check['predicted_worldwide_check'])
        print(f"  R² (проверочный): {r2_check:.4f}")
        print(f"  MSE (проверочный): {mse_check:.2f}")
        print(f"  MAE (проверочный): {mae_check:.2f}")

except Exception as e_check:
    print(f"Ошибка во время финальной проверки: {e_check}")
    import traceback
    traceback.print_exc()

print("\n--- СКРИПТ ЗАВЕРШЕН ---")