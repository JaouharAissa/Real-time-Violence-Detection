"""FastAPI application for violence detection"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
from utils import predict_violence
from config import *

app = FastAPI(title="Violence Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None

@app.on_event("startup")
async def load_model():
    global model
    model_path = os.path.join(MODELS_DIR, 'final_model_two_stream.h5')
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}")
        print("Please train the model first using 02_model_training_optical_flow.ipynb")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Two-Stream model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/")
async def root():
    return {
        "message": "Violence Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict violence in uploaded video"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: mp4, avi, mov, mkv"
        )
    
    # Save uploaded file temporarily
    temp_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Predict
        result = predict_violence(model, temp_path)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Could not process video")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict violence in multiple videos"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type"
            })
            continue
        
        temp_path = os.path.join(TEMP_DIR, file.filename)
        
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            result = predict_violence(model, temp_path)
            
            if result is None:
                results.append({
                    "filename": file.filename,
                    "error": "Could not process video"
                })
            else:
                results.append({
                    "filename": file.filename,
                    **result
                })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
