# model_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Any, Dict
import pandas as pd
import numpy as np
import logging

# Импортируем логику предсказания из predictor.py
from . import predictor # Относительный импорт, так как main.py и predictor.py в одном пакете

# Настройка логгера для FastAPI приложения
logger = logging.getLogger(__name__) # Используем имя текущего модуля
# Уровень и формат будут унаследованы от базовой конфигурации, 
# сделанной в predictor.py или можно настроить отдельно.
# logging.basicConfig(...) лучше делать один раз при старте приложения,
# predictor.py уже это делает.

# Инициализация FastAPI приложения
app = FastAPI(
    title="Movie Box Office Prediction Service",
    description="Сервис для предсказания кассовых сборов фильмов.",
    version="1.0.0"
)

# Модель входных данных для FastAPI (Pydantic)
class MovieDataInput(BaseModel):
    movie_title: Optional[str] = None 
    movie_year: Optional[float] = Field(default=None)
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
    main_actor_4: Optional[str] = None
    genre_1: Optional[str] = None
    genre_2: Optional[str] = None
    genre_3: Optional[str] = None
    genre_4: Optional[str] = None

    # Pydantic v2 style
    @field_validator('*', mode='before')
    def empty_str_to_none_and_pd_na_to_none(cls, value: Any) -> Optional[Any]:
        if value is pd.NA: # Обработка pandas MissingIndicator
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value
        
    class Config:
        str_strip_whitespace = True # Удаляет пробелы по краям строк (Pydantic v2)
        # Для Pydantic v1 было бы: anystr_strip_whitespace = True


class PredictionOutput(BaseModel):
    predicted_gross: float
    currency: str = "USD"

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI Model Service: Попытка инициализации артефактов при старте...")
    try:
        predictor.get_artifacts() # Это загрузит и закэширует артефакты
        logger.info("FastAPI Model Service: Артефакты успешно инициализированы.")
    except Exception as e:
        logger.critical(f"FastAPI Model Service: КРИТИЧЕСКАЯ ОШИБКА при инициализации артефактов: {e}", exc_info=True)
        # В реальном проде можно здесь остановить приложение, если артефакты критичны
        # raise RuntimeError(f"Не удалось загрузить артефакты: {e}")

@app.post("/predict_movie_gross", response_model=PredictionOutput)
async def predict_movie_gross_endpoint(movie_data: MovieDataInput):
    """
    Принимает данные о фильме и возвращает предсказанные кассовые сборы.
    """
    try:
        # Преобразуем Pydantic модель в словарь
        # exclude_none=False важно, чтобы None передавались как None, а не удалялись
        movie_data_dict = movie_data.model_dump(exclude_none=False) 
        logger.debug(f"Получены данные для предсказания: {movie_data_dict}")

        # Некоторые числовые поля могут прийти как None и должны быть np.nan для pandas
        # preprocess_input_data в predictor.py также должен это учитывать, но здесь дублируем для ясности
        # что наш API-контракт передает None, а pandas ожидает np.nan для числовых пропусков.
        for key in ['budget', 'movie_year']: 
            if movie_data_dict[key] is None:
                movie_data_dict[key] = np.nan
        
        predicted_gross_value = predictor.make_prediction(movie_data_dict)
        
        logger.info(f"Предсказание для '{movie_data_dict.get('movie_title', 'N/A')}': {predicted_gross_value}")
        return PredictionOutput(predicted_gross=predicted_gross_value)

    except RuntimeError as e: 
        logger.error(f"Ошибка сервиса (вероятно, артефакты): {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Ошибка сервиса модели: {str(e)}")
    except ValueError as e: 
        logger.error(f"Ошибка входных данных или предсказания: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Ошибка обработки запроса: {str(e)}")
    except Exception as e:
        logger.critical(f"Непредвиденная внутренняя ошибка сервера: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера.")

@app.get("/health", summary="Проверка работоспособности сервиса")
async def health_check():
    """
    Проверяет, что сервис запущен и артефакты модели доступны.
    """
    try:
        predictor.get_artifacts() # Попытка получить артефакты (проверит кэш или загрузит)
        return {"status": "ok", "message": "Сервис модели исправен, артефакты загружены."}
    except Exception as e:
        logger.error(f"Проверка работоспособности провалена: {str(e)}", exc_info=True)
        # Не выбрасываем HTTPException здесь, чтобы health check всегда возвращал 200,
        # но с информацией об ошибке в теле ответа.
        return {"status": "error", "message": f"Ошибка сервиса модели: {str(e)}"}

# Для локального запуска (не используется Docker напрямую, но полезно для отладки)
if __name__ == "__main__":
    import uvicorn
    # Важно: при запуске uvicorn model_service.main:app --reload
    # __name__ будет "model_service.main", а не "__main__"
    # Этот блок if __name__ == "__main__": сработает только если вы запускаете python model_service/main.py
    # Для uvicorn используйте: uvicorn model_service.main:app --host 0.0.0.0 --port 8000 --reload
    logger.info("Запуск FastAPI приложения через uvicorn (если __name__ == '__main__')...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=1)