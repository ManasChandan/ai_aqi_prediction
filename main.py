from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager

import ModelInteractionUtility as ModelInteractionUtility

# Scheduler Function


@asynccontextmanager
async def schedule(app: FastAPI):
    scheduler = BackgroundScheduler()
    trigger = CronTrigger(
        year="*", month="*", day="*", hour="18", minute="18", second="18"
    )
    scheduler.add_job(ModelInteractionUtility.update_model,
                      'cron', second='*/5')
    # scheduler.add_job(
    #     model.update_model,
    #     trigger=trigger
    # )
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)

app = FastAPI(lifespan=schedule)


@app.get("/health")
async def read_root():
    return f"running file current version -- 0.0.1"


@app.get("/getmodel")
async def get_model():
    return f"{ModelInteractionUtility.get_model()}"
