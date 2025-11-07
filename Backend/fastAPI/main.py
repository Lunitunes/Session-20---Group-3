from fastapi import FastAPI, HTTPException, depends, request, status #the fastapi
import os #files
import pickle #pkl
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import time

app = FastAPI()

model = None
MODEL_FILENAME = "trained_model.pkl"

async def startup(app: FastAPI):
    global model
    try:
        BDIR = os.path.dirname(__file__)
        MODEL_PATH = os.path.join(BDIR, MODEL_FILENAME)

        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError(f"No such file or directory for {MODEL_PATH}")
    except Exception as e:
        print(f"Couldn't load model with error: {e}")
