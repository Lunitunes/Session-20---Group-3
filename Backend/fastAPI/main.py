from fastapi import FastAPI, HTTPException, Depends, Request, status
import os
import pandas as pd
import pickle
from pydantic import BaseModel, Field, ValidationError
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import time
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.middleware(
    # frontend communication. Change details for connection.
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
MODEL_FILENAME = "trained_model.pkl"
# THIS FILE SHOULD BE IN THE SAME DIRECTORY AS BACKEND/FASTAPI/MAIN.PY 

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

class InputData(BaseModel):
    proto: float
    service: float
    state: float
    dur: float
    sbytes: float
    dbytes: float
    spkts: float
    dpkts: float
    sttl: float
    dttl: float
    swin: float
    dwin: float
    ct_srv_src: float
    ct_state_ttl: float
    ct_dst_ltm: float
    ct_src_dport_ltm: float
    ct_dst_sport_ltm: float
    ct_dst_src_ltm: float
    is_ftp_login: float
    ct_ftp_cmd: float
    ct_flw_http_mthd: float


