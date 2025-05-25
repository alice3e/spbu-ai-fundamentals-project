# model_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import pandas as pd # Для pd.NA
import numpy as np # Для np.nan

# Импортируем логику предсказания
import predictor # Это наш predictor.py

# Инициализация FastAPI приложения
app = FastAPI(title="Movie Box Office Prediction Service")

# Модель входных данных для FastAPI (Pydantic)
# Должна соответствовать данным, которые ожидает predictor.preprocess_input_data
class MovieDataInput(BaseModel):
    movie_title: Optional[str] = None # Не используется моделью, но может быть в запросе
    movie_year: Optional[float] = Field(default=None) # float, т.к. может быть NaN
    budget: Optional[float] = Field(default=None)
    run_time: Optional[str] = None
    mpaa: Optional[str] = None
    director: Optional[str] = None
    writer: Optional[str] = None
    producer: Optional[str] = None
    composer: Optional[str] = None
    cinematographer: Optional[str] = None
    distributor: Optional[str] = None
    main_actor_1: Optional[str] = None
    main_actor_2: Optional[str] = None
    main_actor_3: Optional[str] = None
    main_actor_4: Optional[str] = None # pd.NA не сериализуется в JSON, используем None
    genre_1: Optional[str] = None
    genre_2: Optional[str] = None
    genre_3: Optional[str] = None
    genre_4: Optional[str] = None

    # Pydantic v2: from pydantic import field_validator
    # @field_validator('main_actor_4', 'genre_2', 'genre_3', 'genre_4', mode='before')
    @validator('main_actor_4', 'genre_2', 'genre_3', 'genre_4', pre=True, always=True)
    def replace_pd_na_with_none(cls, value):
        if value is pd.NA:
            return None
        return value

    # Валидатор для замены пустых строк на None, если они приходят
    # Pydantic v2: @field_validator('*', mode='before')
    @validator('*', pre=True, always=True)
    def empty_str_to_none(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value
        
    class Config:
        # Pydantic v1
        # anystr_strip_whitespace = True # Удаляет пробелы по краям строк
        # Pydantic v2
        str_strip_whitespace = True


class PredictionOutput(BaseModel):
    predicted_gross: float

# Загрузка артефактов при старте сервиса (ленивая загрузка при первом вызове get_artifacts)
@app.on_event("startup")
async def startup_event():
    print("FastAPI Model Service: Попытка инициализации артефактов...")
    try:
        predictor.get_artifacts() # Это загрузит и закэширует артефакты
        print("FastAPI Model Service: Артефакты успешно инициализированы.")
    except Exception as e:
        print(f"FastAPI Model Service: КРИТИЧЕСКАЯ ОШИБКА при инициализации артефактов: {e}")
        # Можно остановить приложение, если артефакты не загрузятся
        # import sys
        # sys.exit(1)

@app.post("/predict_movie_gross", response_model=PredictionOutput)
async def predict_movie_gross_endpoint(movie_data: MovieDataInput):
    try:
        # Преобразуем Pydantic модель в словарь, который ожидает make_prediction
        movie_data_dict = movie_data.model_dump(exclude_none=False) # Pydantic v2
        # movie_data_dict = movie_data.dict(exclude_none=False) # Pydantic v1

        # Заменяем None на np.nan для числовых полей, где это ожидается Pandas
        # (preprocess_input_data уже должен это делать, но для надежности)
        for key in ['budget', 'movie_year']: # и другие числовые, если есть
            if movie_data_dict[key] is None:
                movie_data_dict[key] = np.nan
        
        # Для категориальных полей, где None/NaN обрабатывается как "Unknown" или модой,
        # None из Pydantic (бывший пустой ввод) подходит.
        # predictor.preprocess_input_data позаботится о pd.NA/np.nan

        predicted_gross = predictor.make_prediction(movie_data_dict)
        return PredictionOutput(predicted_gross=predicted_gross)
    except RuntimeError as e: # Ошибки, связанные с загрузкой артефактов
        raise HTTPException(status_code=503, detail=f"Ошибка сервиса модели: {str(e)}")
    except ValueError as e: # Ошибки валидации данных или внутри предсказания
        raise HTTPException(status_code=400, detail=f"Ошибка входных данных: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка в эндпоинте /predict_movie_gross: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/health")
async def health_check():
    # Простая проверка работоспособности
    try:
        # Попытка получить артефакты (проверит, что они загружены или загрузит)
        predictor.get_artifacts()
        return {"status": "ok", "message": "Model service is healthy and artifacts are loaded."}
    except Exception as e:
        return {"status": "error", "message": f"Model service error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # Для локального запуска: uvicorn model_service.main:app --reload --port 8000
    # Docker будет использовать команду из Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8017)