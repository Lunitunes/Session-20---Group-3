from fastapi import FastAPI, HTTPException, UploadFile, File, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import pandas as pd
import joblib
from pydantic import BaseModel
from pathlib import Path
import uuid
from datetime import datetime
import json

"""
Imports:
- FastAPI framework:
          HTTPException - Handling HTTP responses.
          UploadFile - Handling CSV file uploads to the API.
          File - Secondary handling CSV file uploads.
          Status - Custom error-checking, validation and error handling status 
          Form - Input fields and form validation.
- CorsMiddleware
          A component of the FastAPI Library used for configuring the origin that we want to permit to access the API.
- JSONResponse
          A component of the FastAPI Library that is used for returning a JSON-encoded HTTP response.
- OS Import
          A standard import for filepath manipulation.
- Pandas
          A non-standard import used for data analysis.
- Pydantic
          An import used for data validation.
- Pathlib
          An import for handling and manipulating filepaths.
- UUID
          An import for generating universally unique identifiers
- datetime
          An import for date/time representation
- json
          An import for reading and writing .JSON files
"""

app = FastAPI()
 
app.add_middleware(
    # Allows for FastAPI to accept requests from any origin.
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
BASE_PATH = Path(__file__).resolve().parent

model = None
MODEL_FILENAME = "trained_model.pkl"
# Pickle file containing our trained model - Random Forest Classifier

ANALYSIS_PATH = BASE_PATH / "analysis_data"
ANALYSIS_PATH.mkdir(exist_ok=True)

INDEX_PATH    = ANALYSIS_PATH / "index.json"
if not INDEX_PATH.exists():
    INDEX_PATH.write_text("[]")
 
@app.on_event("startup")
# Loads the Pickle/.PKL file
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
 # Category map for mapping data predictions.
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
# This function allows for the upload of a CSV file and generates a UUID. This UUID is stored inside of index.JSON.
async def upload_csv(name: str = Form(...) ,file: UploadFile = File()):
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "Only CSV files are supported."}
         # Error handling for CSV
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

        df = df.dropna(how="all")

        if len(df) != 1:
            raise ValueError("CSV file must contain exactly one row of data.")
                  # This is not inclusive of the header data
              
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=["encodedcategory"], errors="ignore")
        df = df[model.feature_names_in_]

        predictions = model.predict(df)
        df["prediction"] = predictions
        CATEGORY_MAPPING = [CATEGORY_MAP.get(int(p), "Unknown") for p in predictions]

        analysis_id = uuid.uuid4().hex[:8]
        # Saves Analysis Metadata to JSON
        record = {
          "analysis_id" : analysis_id,
          "analysis_name" : name,
          "row_count" : len(df),
          "timestamp" : datetime.now().isoformat(),
          "predicted_type" : CATEGORY_MAPPING
        }
        
        CSV_PATH = ANALYSIS_PATH / f"{analysis_id}.csv"
        df.to_csv(CSV_PATH, index=False)

        index = json.loads(INDEX_PATH.read_text())
        index.append(record)
        INDEX_PATH.write_text(json.dumps(index, indent=2))

        return JSONResponse(record)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error processing file: {str(e)}"}
        )

@app.get("/return_index_ids")
async def return_index_ids():
    try:
        index = json.loads(INDEX_PATH.read_text())
        ids = [record["analysis_id"] for record in index]
        return JSONResponse(ids)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error retrieving index IDs: {str(e)}"}
        )

@app.get("/return_index")
# This function obtains and displays the contents of index.JSON.
async def return_index():
    try:
        index = json.loads(INDEX_PATH.read_text())
        return index
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error retrieving index: {str(e)}"}
        )

@app.get("/get_csv/{analysis_id}")
# This functions obtains a previously analyzed CSV by taking its generated UUID as a parameter.
async def get_csv(analysis_id: str):
    CSV_PATH = ANALYSIS_PATH / f"{analysis_id}.csv"
    
    if not CSV_PATH.exists():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Analysis file not found."}
        )
    try:
        df = pd.read_csv(CSV_PATH)
        data = df.to_dict(orient="records")
        
        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error reading CSV: {str(e)}"}
        )

TRAINING_DATA_PATH = BASE_PATH / "training_data.csv"
# ID
@app.get("/get_training_data{training_data_id}")
async def get_training_data(training_data_id: str):
    csv_file = TRAINING_DATA_PATH / f"{training_data_id}.csv"
          
    if not csv_file.exists():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Training data file not found."}
        )
    try:
        df = pd.read_csv(csv_file)
        data = df.to_dict(orient="records")
        
        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error reading training data CSV: {str(e)}"}
        )
              
@app.delete("/delete_analysis/{analysis_id}")
# This function deletes the analysis csv file by its UUID.
async def delete_analysis(analysis_id: str):
    CSV_PATH = ANALYSIS_PATH / f"{analysis_id}.csv"
    try:
        if CSV_PATH.exists():
            os.remove(CSV_PATH)
        
        index = json.loads(INDEX_PATH.read_text())
        index = [record for record in index if record["analysis_id"] != analysis_id]
        INDEX_PATH.write_text(json.dumps(index, indent=2))

        return JSONResponse(
            content={"message": f"Analysis {analysis_id} deleted successfully."}
        )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error deleting analysis: {str(e)}"}
        )
