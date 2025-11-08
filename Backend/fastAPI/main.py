from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import pandas as pd
import joblib
from pydantic import BaseModel
 
app = FastAPI()
 
app.add_middleware(
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
 
@app.on_event("startup")
async def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded successfully: {type(model)}")
        else:
            print(f"Model file not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
 
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

CATEGORY_MAP = {
    0: "Normal",
    1: "Fuzzers",
    2: "Analysis",
    3: "Backdoor",
    4: "DoS",
    5: "Exploits",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms"
}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File()):
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Only CSV files are supported."}
        )
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"message": "Model is not loaded."}
        )
    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=["encodedcategory"], errors="ignore")
        df = df[model.feature_names_in_]

        predictions = model.predict(df)
        CATEGORY_MAPPING = [CATEGORY_MAP.get(int(p), "Unknown") for p in predictions]

        return JSONResponse(content={
            "predictions": CATEGORY_MAPPING,
            "row_count": len(df)
        })
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error processing file: {str(e)}"}
        )
 
@app.post("/predict")
async def predict(input_data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
 
    input_df = pd.DataFrame([input_data.dict()])
    try:
        prediction = model.predict(input_df)

        category_mapping = CATEGORY_MAP.get(int(prediction[0]), "Unknown")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
 
    return {"prediction": prediction[0]}
