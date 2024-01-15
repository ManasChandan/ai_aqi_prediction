import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, validator
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager

import ModelInteractionUtility as ModelInteractionUtility

# Validator
class InputData(BaseModel):
    date: str
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    nh3: float

# Scheduler Function


@asynccontextmanager
async def schedule(app: FastAPI):
    scheduler = BackgroundScheduler()
    trigger =  CronTrigger.from_crontab("05 19 * * *")
    delete_trigger = CronTrigger.from_crontab("45 19 * * *")
    scheduler.add_job(ModelInteractionUtility.model_status,
                      'cron', second='*/20')
    scheduler.add_job(
        ModelInteractionUtility.update_model, trigger
    )
    scheduler.add_job(
        ModelInteractionUtility.delete_model_from_db,delete_trigger
    )
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)

app = FastAPI(lifespan=schedule)


@app.get("/health")
async def read_root():
    return f"running file current version -- 0.0.3"

@app.post("/predict")
async def predict(data: InputData):
    try:
        prediction = ModelInteractionUtility.predict(data.model_dump())
        return JSONResponse(content=prediction)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))