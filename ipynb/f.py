import pandas as pd
import random
import numpy as np

random.seed(42)
np.random.seed(42)

data = pd.read_csv("all_data.csv")
data.head()

data.columns

'''
Index(['movie_id', 'movie_title', 'movie_year', 'director', 'writer',
       'producer', 'composer', 'cinematographer', 'main_actor_1',
       'main_actor_2', 'main_actor_3', 'main_actor_4', 'budget', 'domestic',
       'international', 'worldwide', 'mpaa', 'run_time', 'genre_1', 'genre_2',
       'genre_3', 'genre_4', 'link', 'distributor'],
      dtype='object')
'''

nan_data = (data.isnull().mean() * 100).reset_index()
nan_data.columns = ["column_name", "percentage"]
nan_data.sort_values("percentage", ascending=False, inplace=True)
nan_data.head(24)
'''
	column_name	percentage
21	genre_4	64.574226
23	distributor	43.556566
20	genre_3	30.914496
19	genre_2	8.078335
14	international	7.169086
16	mpaa	3.899283
7	cinematographer	3.567057
6	composer	3.112432
13	domestic	0.332226
5	producer	0.314740
4	writer	0.227312
15	worldwide	0.122399
11	main_actor_4	0.052457
1	movie_title	0.000000
2	movie_year	0.000000
'''

data = data.dropna(subset=['worldwide'])

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
test_data.to_csv("test.csv", index=False)

train_movie_title = train_data['movie_title']
columns_to_drop = ['movie_id', 'movie_title', 'link']
#columns_to_drop = ['movie_id', 'link']
train_data = train_data.drop(columns=columns_to_drop)
train_data.head()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="whitegrid")


plt.rcParams['figure.figsize'] = (12, 6)
sns.set_palette("husl")

# 1. Heatmap корреляций числовых признаков
# Выбираем только числовые колонки
numeric_cols = train_data.select_dtypes(include=[np.number]).columns

# Считаем корреляции и сортируем по целевой переменной
corr_matrix = train_data[numeric_cols].corr()
target_corr = corr_matrix['worldwide'].sort_values(ascending=False)

# Визуализируем топ-20 признаков по корреляции
plt.figure(figsize=(12, 10))
top_features = target_corr.index[1:21]  # исключаем сам worldwide
sns.heatmap(train_data[top_features].corr(), 
            annot=True, fmt=".2f", 
            cmap='coolwarm', 
            center=0,
            vmin=-1, vmax=1,
            linewidths=0.5)
plt.title('Тепловая карта корреляций (топ-20 признаков)', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Распределение целевой переменной
plt.figure(figsize=(12, 6))

# Гистограмма
plt.subplot(1, 2, 1)
sns.histplot(train_data['worldwide'], kde=True, bins=50)
plt.title(f'Распределение worldwide\nSkewness: {train_data["worldwide"].skew():.2f}')
plt.xlabel('Цена ($)')
plt.ylabel('Частота')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=train_data['worldwide'])
plt.title('Boxplot worldwide')
plt.ylabel('Цена ($)')

plt.suptitle('Анализ целевой переменной', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

for col in ["director", "writer", "producer", "composer", "cinematographer"]:
    counts = train_data[col].value_counts()
    train_data[f"{col}_experience"] = train_data[col].map(counts)

# опыт главных актёров и суммарная "звёздность"
for i in range(1, 5):
    col = f"main_actor_{i}"
    counts = train_data[col].value_counts()
    train_data[f"{col}_experience"] = train_data[col].map(counts)
train_data["cast_popularity"] = sum(train_data[f"main_actor_{i}_experience"] for i in range(1, 5))

import re

def convert_runtime_to_minutes(value):
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None  # если формат не распознан

train_data['run_time'] = train_data['run_time'].apply(convert_runtime_to_minutes)

from sklearn.impute import SimpleImputer

num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy='median')
train_data[num_cols] = num_imputer.fit_transform(train_data[num_cols])

nan_data = (train_data.isnull().mean() * 100).reset_index()
nan_data.columns = ["column_name", "percentage"]
nan_data.sort_values("percentage", ascending=False, inplace=True)
nan_data.head(24)

'''
	column_name	percentage
19	genre_4	64.532266
20	distributor	42.921461
18	genre_3	31.015508
17	genre_2	8.054027
14	mpaa	3.851926
'''

genre_cols = ["genre_1", "genre_2", "genre_3", "genre_4"]
cols_to_fill = ['cinematographer', 'composer', 'producer', 'writer']

for col in cols_to_fill:
    # Группировка по distributor и заполнение пропусков модой по группе
    train_data[col] = train_data.groupby('director')[col]\
        .transform(lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
        
columns_to_fill = ['genre_2', 'genre_3', 'genre_4', 'main_actor_4']
train_data[columns_to_fill] = train_data[columns_to_fill].fillna('Unknown')
cat_cols = train_data.select_dtypes(include=['object']).columns

cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[cat_cols] = cat_imputer.fit_transform(train_data[cat_cols])
nan_data = (train_data.isnull().mean() * 100).reset_index()
nan_data.columns = ["column_name", "percentage"]
nan_data.sort_values("percentage", ascending=False, inplace=True)
nan_data.head(24)


# Вывод столбцов и колисество значений с ними
cat_cols = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

cat_stats = pd.DataFrame({
    'Column': cat_cols,
    'Unique_Count': [train_data[col].nunique() for col in cat_cols],
    'Unique_Values': [train_data[col].unique() for col in cat_cols]
})

cat_stats = cat_stats.sort_values(by='Unique_Count', ascending=False)
print(cat_stats[['Column', 'Unique_Count']])

columns_for_target_encoding = ['main_actor_4', 'main_actor_3', 'writer', 'main_actor_2', 'producer', 'director', 'main_actor_1', 'cinematographer', 'composer', 'distributor']
columns_for_one_hot = ['genre_2', 'genre_3', 'genre_4', 'genre_1', 'mpaa']

train_data.columns
'''
Index(['movie_year', 'director', 'writer', 'producer', 'composer',
       'cinematographer', 'main_actor_1', 'main_actor_2', 'main_actor_3',
       'main_actor_4', 'budget', 'domestic', 'international', 'worldwide',
       'mpaa', 'run_time', 'genre_1', 'genre_2', 'genre_3', 'genre_4',
       'distributor', 'director_experience', 'writer_experience',
       'producer_experience', 'composer_experience',
       'cinematographer_experience', 'main_actor_1_experience',
       'main_actor_2_experience', 'main_actor_3_experience',
       'main_actor_4_experience', 'cast_popularity'],
      dtype='object')
'''

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_data[columns_for_one_hot])

# Трансформируем трейн и тест
train_encoded = ohe.transform(train_data[columns_for_one_hot])
#train_data = pd.get_dummies(train_data, columns=columns_for_one_hot)
train_encoded_df = pd.DataFrame(
    train_encoded,
    columns=ohe.get_feature_names_out(columns_for_one_hot),
    index=train_data.index
)
train_data = train_data.drop(columns_for_one_hot, axis=1).join(train_encoded_df)
train_data.columns

import category_encoders as ce
encoder = ce.TargetEncoder(cols=columns_for_target_encoding)
train_data_encoded = encoder.fit_transform(train_data[columns_for_target_encoding], train_data['worldwide'])
train_data[columns_for_target_encoding] = train_data_encoded

X = train_data.drop("worldwide", axis=1)
y = train_data["worldwide"]
#y = np.log1p(train_data["worldwide"])

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth':    [4, 6, 8],
    'learning_rate':[0.01, 0.05, 0.1],
    'subsample':    [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha':    [0, 0.5, 1],
    'reg_lambda':   [1, 2, 5]
}

# 4) Создаём XGB с фиксированным seed и детерминированным методом
base_model = XGBRegressor(
    random_state=42,
    tree_method='hist',            # более детерминированный
    enable_categorical=False,
    n_jobs=1
)

# 5) RandomizedSearchCV с фиксированным seed
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=1,              # лучше 1, чтобы исключить недетерминированность
    random_state=42
)

# 6) Обучаем
random_search.fit(X_train, y_train)

# 7) Оцениваем
best = random_search.best_estimator_
y_pred = best.predict(X_val)
print(f"R²:  {r2_score(y_val, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_val, y_pred):.2f}")
#Best Params: {'subsample': 0.8, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05, 'colsample_bytree': 1.0}
#Best R² Score on Test: 0.9990
#Best MSE on Test: 37542782216082.52

#Best R² Score on Test: 0.9990
#Best MSE on Test: 36496695202244.70

#Best R² Score on Test: 0.9991
#Best MSE on Test: 32663862493933.65


#R²:  0.9993
#MSE: 28215564467164.59


# TEST DATA
test_data = pd.read_csv('test.csv')
test_movie_title = test_data['movie_title']
columns_to_drop = ['movie_id', 'movie_title', 'link']
#columns_to_drop = ['movie_id', 'link']
test_data = test_data.drop(columns=columns_to_drop)

for col in ["director", "writer", "producer", "composer", "cinematographer"]:
    counts = train_data[col].value_counts()
    test_data[f"{col}_experience"] = test_data[col].map(counts).fillna(0)
for i in range(1, 5):
    col = f"main_actor_{i}"
    counts = train_data[col].value_counts()
    test_data[f"{col}_experience"] = test_data[col].map(counts).fillna(0)

test_data["cast_popularity"] = sum(test_data[f"main_actor_{i}_experience"] for i in range(1, 5))
test_data.head()

def convert_runtime_to_minutes(value):
    match = re.match(r'(?:(\d+)\s*hr)?\s*(?:(\d+)\s*min)?', str(value))
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None  # если формат не распознан

test_data['run_time'] = test_data['run_time'].apply(convert_runtime_to_minutes)

test_num_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
test_data[test_num_cols] = num_imputer.transform(test_data[test_num_cols])


cols_to_fill = ['cinematographer', 'composer', 'producer', 'writer']

for col in cols_to_fill:
    mode_values = train_data.groupby('director')[col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    test_data[col] = test_data.apply(lambda row: mode_values.get(row['director'], np.nan) if pd.isnull(row[col]) else row[col], axis=1)


columns_to_fill = ['genre_2', 'genre_3', 'genre_4', 'main_actor_4']
test_data[columns_to_fill] = test_data[columns_to_fill].fillna('Unknown')

cat_cols = test_data.select_dtypes(include=['object']).columns

test_data[cat_cols] = cat_imputer.fit_transform(test_data[cat_cols])

nan_data = (test_data.isnull().mean() * 100).reset_index()
nan_data.columns = ["column_name", "percentage"]
nan_data.sort_values("percentage", ascending=False, inplace=True)
nan_data.head(24)


test_encoded = ohe.transform(test_data[columns_for_one_hot])
test_encoded_df = pd.DataFrame(
    test_encoded,
    columns=ohe.get_feature_names_out(columns_for_one_hot),
    index=test_data.index
)

test_data = test_data.drop(columns_for_one_hot, axis=1).join(test_encoded_df)
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
test_data.columns


test_data_encoded = encoder.fit_transform(test_data[columns_for_target_encoding], test_data['worldwide'])
test_data[columns_for_target_encoding] = test_data_encoded

X_test =test_data.drop("worldwide", axis=1)
y_test = test_data["worldwide"]

X_test = scaler.transform(X_test)

y_pred = best.predict(X_test)
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

'''
y_pred = best.predict(X_test)
print(f"R²:  {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
'''

