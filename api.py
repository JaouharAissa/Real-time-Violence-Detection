"""
Violence Detection API
FastAPI server for video violence detection
"""

import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import tempfile
import json
import zipfile
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from db import SessionLocal, engine
from models import Alert
from schemas import DashboardStats, HourlyActivityPoint, to_alert_item
from notifier import send_violence_email

class PatchedBatchNormalization(keras.layers.BatchNormalization):
    """Compatibility shim for models saved with legacy BatchNorm renorm args."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)

class PatchedDense(keras.layers.Dense):
    """Compatibility shim for models saved with Dense quantization_config."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


def _strip_incompatible_keys(obj):
    if isinstance(obj, dict):
        obj.pop("renorm", None)
        obj.pop("renorm_clipping", None)
        obj.pop("renorm_momentum", None)
        obj.pop("quantization_config", None)
        for v in list(obj.values()):
            _strip_incompatible_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_incompatible_keys(v)


def _patch_keras_file_remove_renorm(src_path: str) -> str:
    """Create a patched copy of a .keras file removing incompatible keys."""
    patched_path = os.path.join(
        tempfile.gettempdir(),
        f"patched_{os.path.basename(src_path)}",
    )

    with zipfile.ZipFile(src_path, "r") as zin:
        with zipfile.ZipFile(patched_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith("config.json"):
                    cfg = json.loads(data.decode("utf-8"))
                    _strip_incompatible_keys(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(item, data)

    return patched_path

# FastAPI app (do not use root_path here unless the proxy *rewrites* paths).
app = FastAPI(title="Violence Detection API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    # Temporary: allow all origins while validating GCP ingress + frontend URLs.
    # Tighten this later to your real frontend domain(s).
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB init
from db import Base  # noqa: E402

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load model
MODEL_PATH = "models/phase2_20260415_184926.keras"
model = None
model_load_error = None
try:
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "BatchNormalization": PatchedBatchNormalization,
                "Dense": PatchedDense,
            },
            safe_mode=False,
        )
    except Exception as e:
        patched_path = _patch_keras_file_remove_renorm(MODEL_PATH)
        print(f"Retrying with patched model file: {patched_path}")
        model = keras.models.load_model(
            patched_path,
            compile=False,
            custom_objects={
                "BatchNormalization": PatchedBatchNormalization,
                "Dense": PatchedDense,
            },
            safe_mode=False,
        )
    model_load_error = None
    print(f"Model loaded: {model.count_params():,} parameters")
except Exception as e:
    model_load_error = str(e)
    print(f"Model load failed: {model_load_error}")

SEQUENCE_LENGTH = 30
FRAME_HEIGHT = 128
FRAME_WIDTH = 128

def extract_frames_and_flow(video_path, max_frames=SEQUENCE_LENGTH):
    """Extract RGB frames and optical flow from video"""
    rgb_frames = []
    flow_frames = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None, None
    
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    prev_gray = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # RGB
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        rgb_frames.append(frame_rgb)
        
        # Optical Flow
        curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = np.clip(magnitude / 20.0, 0, 1)
            angle_norm = angle / (2 * np.pi)
            flow_representation = np.stack([magnitude, angle_norm], axis=-1).astype(np.float32)
            flow_frames.append(flow_representation)
        
        prev_gray = curr_gray
    
    cap.release()
    
    if len(rgb_frames) < max_frames * 0.5:
        return None, None
    
    # Padding
    if len(flow_frames) > 0:
        flow_frames.insert(0, flow_frames[0].copy())
    
    while len(rgb_frames) < max_frames:
        rgb_frames.append(rgb_frames[-1].copy())
    
    while len(flow_frames) < max_frames:
        if len(flow_frames) > 0:
            flow_frames.append(flow_frames[-1].copy())
        else:
            flow_frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), dtype=np.float32))
    
    return np.array(rgb_frames[:max_frames]), np.array(flow_frames[:max_frames])


@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Violence Detection API",
        "version": "1.0",
        "model": "Two-Stream (ResNet50 + Optical Flow + LSTM)",
        "accuracy": "80.34%",
        "endpoints": {
            "/predict": "POST - Upload video for violence detection",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_load_error": model_load_error,
    }


@app.get("/alerts/recent")
def alerts_recent(limit: int = 10):
    with SessionLocal() as db:
        rows = (
            db.execute(select(Alert).order_by(Alert.created_at.desc()).limit(max(1, min(limit, 50))))
            .scalars()
            .all()
        )
        return [to_alert_item(a).model_dump() for a in rows]


@app.get("/alerts")
def alerts_list(
    page: int = 1,
    pageSize: int = 10,
    from_: str | None = None,
    to: str | None = None,
    label: str | None = None,
):
    try:
        page = max(1, int(page))
        page_size = max(1, min(50, int(pageSize)))
    except Exception:
        page = 1
        page_size = 10

    def parse_date(s: str) -> datetime:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    with SessionLocal() as db:
        q = select(Alert)
        cq = select(func.count()).select_from(Alert)

        if label:
            q = q.where(Alert.label == label)
            cq = cq.where(Alert.label == label)

        if from_:
            dt = parse_date(from_)
            q = q.where(Alert.created_at >= dt.replace(tzinfo=None))
            cq = cq.where(Alert.created_at >= dt.replace(tzinfo=None))

        if to:
            dt = parse_date(to) + timedelta(days=1)
            q = q.where(Alert.created_at < dt.replace(tzinfo=None))
            cq = cq.where(Alert.created_at < dt.replace(tzinfo=None))

        total = int(db.execute(cq).scalar() or 0)
        rows = (
            db.execute(
                q.order_by(Alert.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
            )
            .scalars()
            .all()
        )

        return {
            "items": [to_alert_item(a).model_dump() for a in rows],
            "total": total,
        }


@app.get("/dashboard/stats")
def dashboard_stats():
    with SessionLocal() as db:
        total = int(db.execute(select(func.count()).select_from(Alert)).scalar() or 0)
        violent = int(
            db.execute(
                select(func.count()).select_from(Alert).where(Alert.label == "Violence")
            ).scalar()
            or 0
        )
        violence_rate = float(violent / total) if total else 0.0

        now = datetime.utcnow()
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)

        today_rows = (
            db.execute(select(Alert).where(Alert.created_at >= start).where(Alert.created_at < end))
            .scalars()
            .all()
        )
        alerts_today = sum(1 for a in today_rows if a.label == "Violence")

        buckets = {h: {"total": 0, "violent": 0} for h in range(24)}
        for a in today_rows:
            h = a.created_at.hour
            buckets[h]["total"] += 1
            if a.label == "Violence":
                buckets[h]["violent"] += 1

        activity = [
            HourlyActivityPoint(hour=h, total=buckets[h]["total"], violent=buckets[h]["violent"])
            for h in range(24)
        ]

        recent = (
            db.execute(select(Alert).order_by(Alert.created_at.desc()).limit(8)).scalars().all()
        )

        stats = DashboardStats(
            totalAnalyses=total,
            violenceRate=violence_rate,
            alertsToday=alerts_today,
            activityByHour=activity,
            recentAlerts=[to_alert_item(a) for a in recent],
        )
        return stats.model_dump()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict violence in uploaded video

    Returns:
        - prediction: "Violence" or "Non-Violence"
        - confidence: probability score
        - violence_probability: probability of violence
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_load_error}")

    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Use mp4, avi, mov, or mkv")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        rgb_frames, flow_frames = extract_frames_and_flow(tmp_path)
        
        if rgb_frames is None or flow_frames is None:
            raise HTTPException(status_code=400, detail="Could not process video")
        
        rgb_batch = np.expand_dims(rgb_frames, axis=0)
        flow_batch = np.expand_dims(flow_frames, axis=0)
        
        prediction = model.predict([rgb_batch, flow_batch], verbose=0)
        violence_prob = float(prediction[0][0])
        
        result = {
            "filename": file.filename,
            "prediction": "Violence" if violence_prob > 0.5 else "Non-Violence",
            "confidence": float(max(violence_prob, 1 - violence_prob)),
            "violence_probability": violence_prob,
            "non_violence_probability": 1 - violence_prob
        }

        try:
            with SessionLocal() as db:
                alert = Alert(
                    created_at=datetime.utcnow(),
                    source_type="upload",
                    source_ref=file.filename,
                    label=result["prediction"],
                    confidence=float(result["confidence"]),
                    violence_probability=float(violence_prob),
                )
                db.add(alert)
                db.commit()

                if alert.label == "Violence":
                    send_violence_email(
                        subject="ALERTE: Violence détectée",
                        body=(
                            "Une alerte 'Violence' a été détectée.\n\n"
                            f"- Source: upload\n"
                            f"- Fichier: {file.filename}\n"
                            f"- Confidence: {float(result['confidence']):.3f}\n"
                            f"- Violence probability: {violence_prob:.3f}\n"
                            f"- Date (UTC): {datetime.utcnow().isoformat()}Z\n"
                        ),
                    )
        except Exception:
            pass
        
        return JSONResponse(content=result)
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Violence Detection API")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Accuracy: 80.34%")
    print("\nAPI will be available at:")
    print("  - http://localhost:8000")
    print("  - http://localhost:8000/docs (Swagger UI)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)