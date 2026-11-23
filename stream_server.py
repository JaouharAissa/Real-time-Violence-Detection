"""
Socket.IO stream server (OpenCV) for the React Live Stream page.

- Listens on :8001
- Client emits: "start" { source: "0" | url, threshold?: number }
- Server emits: "frame" { frame, label, confidence, ts }

This is a pragmatic MVP to validate end-to-end streaming with a PC webcam.
"""

import base64
import os
import asyncio
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import socketio
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI
from db import SessionLocal
from models import Alert
from notifier import send_violence_email


MODEL_PATH = os.environ.get("MODEL_PATH", "models/phase2_20260415_184926.keras")
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", "30"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "128"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "128"))
ALERT_COOLDOWN_S = float(os.environ.get("ALERT_COOLDOWN_S", "60"))
# Default matches local dev. In GKE behind path-based ingress, set SOCKETIO_PATH=/stream/socket.io
SOCKETIO_PATH = os.environ.get("SOCKETIO_PATH", "/socket.io")


class PatchedBatchNormalization(keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)


class PatchedDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


def _encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_sequence_and_flow(frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb_frames: list[np.ndarray] = []
    flow_frames: list[np.ndarray] = []

    prev_gray: Optional[np.ndarray] = None
    for frame in frames_bgr:
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_frames.append(frame_rgb)

        curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = np.clip(magnitude / 20.0, 0, 1)
            angle_norm = angle / (2 * np.pi)
            flow_representation = np.stack([magnitude, angle_norm], axis=-1).astype(np.float32)
            flow_frames.append(flow_representation)
        prev_gray = curr_gray

    if flow_frames:
        flow_frames.insert(0, flow_frames[0].copy())
    while len(flow_frames) < len(rgb_frames):
        flow_frames.append(flow_frames[-1].copy() if flow_frames else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 2), np.float32))

    rgb = np.array(rgb_frames[-SEQUENCE_LENGTH:], dtype=np.float32)
    flow = np.array(flow_frames[-SEQUENCE_LENGTH:], dtype=np.float32)
    return rgb, flow


def _load_model() -> Any:
    # Try normal load; if it fails due to incompatibilities, use the same patched approach as api.py.
    try:
        return keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"BatchNormalization": PatchedBatchNormalization, "Dense": PatchedDense},
            safe_mode=False,
        )
    except Exception:
        # Fallback: rely on api.py (which patches .keras) by importing it and using its patched loader if present.
        # This keeps stream server aligned with the backend's compatibility logic.
        import importlib

        api = importlib.import_module("api")
        if getattr(api, "model", None) is not None:
            return api.model
        raise


MODEL = _load_model()


sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Expose plain HTTP health endpoints so GCLB checks get a 200.
app = FastAPI(title="Violence Stream Server", version="1.0")


@app.get("/")
def root():
    return {"service": "stream", "status": "ok"}


@app.get("/health")
def health():
    return {"service": "stream", "status": "healthy"}


# When the app is served behind a path-based ingress (e.g. `/stream`),
# requests arrive with that prefix and would otherwise 404. Support both.
@app.get("/stream")
def root_stream():
    return {"service": "stream", "status": "ok"}


@app.get("/stream/health")
def health_stream():
    return {"service": "stream", "status": "healthy"}


# Wrap FastAPI app with Socket.IO app. 
# This intercepts requests starting with socketio_path before they reach FastAPI.
sio_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/stream/socket.io")

EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


class StreamWorker:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sid: Optional[str] = None
        self._source: str = "0"
        self._threshold: float = 0.7

        self.streams = {}  # sid -> {"thread": thread, "stop_event": event, "namespace": ns}

    def start(self, sid: str, source: str, threshold: float, namespace: str = "/"):
        with self._lock:
            if sid in self.streams:
                self.stop(sid)
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(sid, source, threshold, namespace, stop_event), daemon=True)
            self.streams[sid] = {"thread": thread, "stop_event": stop_event, "namespace": namespace}
            thread.start()

    def stop(self, sid: str):
        with self._lock:
            if sid in self.streams:
                self.streams[sid]["stop_event"].set()
                del self.streams[sid]

    def _run(self, sid, source, threshold, namespace, stop_event):
        try:
            cap_src = int(source) if str(source).isdigit() else source
        except Exception:
            cap_src = source

        cap = cv2.VideoCapture(cap_src)
        if not cap.isOpened():
            if EVENT_LOOP is not None:
                asyncio.run_coroutine_threadsafe(
                    sio.emit("status", {"running": False, "error": f"Cannot open camera: {source}"}, to=sid, namespace=namespace),
                    EVENT_LOOP,
                )
            return

        buffer = deque(maxlen=SEQUENCE_LENGTH)
        last_pred_ts = 0.0
        last_label = "—"
        last_conf = 0.0
        fps_limit = 15.0
        min_dt = 1.0 / fps_limit
        last_frame_time = 0.0
        last_violence_alert_ts = 0.0

        while not stop_event.is_set():
            now = time.time()
            if now - last_frame_time < min_dt:
                time.sleep(0.001)
                continue
            last_frame_time = now

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            buffer.append(frame)
            if len(buffer) >= SEQUENCE_LENGTH and (now - last_pred_ts) > 1.0:
                last_pred_ts = now
                try:
                    rgb, flow = _extract_sequence_and_flow(list(buffer))
                    pred = MODEL.predict([np.expand_dims(rgb, 0), np.expand_dims(flow, 0)], verbose=0)
                    violence_prob = float(pred[0][0])
                    last_conf = float(max(violence_prob, 1.0 - violence_prob))
                    last_label = "Violence" if violence_prob > threshold else "Non-Violence"
                    if last_label == "Violence":
                        if now - last_violence_alert_ts >= max(0.0, ALERT_COOLDOWN_S):
                            last_violence_alert_ts = now
                            self._save_violence_alert(source=source, confidence=last_conf, violence_prob=violence_prob)
                except Exception:
                    pass

            payload = {
                "image": f"data:image/jpeg;base64,{_encode_jpeg(frame)}",
                "label": last_label,
                "confidence": float(last_conf),
                "ts": datetime.utcnow().isoformat() + "Z",
            }

            if EVENT_LOOP is not None:
                try:
                    future = asyncio.run_coroutine_threadsafe(sio.emit("frame", payload, to=sid, namespace=namespace), EVENT_LOOP)
                    future.result(timeout=2.0)
                except Exception as e:
                    print(f"[WORKER] Emission failed for sid={sid}: {e}")
                    break

        cap.release()

    def _save_violence_alert(self, *, source: str, confidence: float, violence_prob: float):
        try:
            with SessionLocal() as db:
                db.add(Alert(created_at=datetime.utcnow(), source_type="live", source_ref=str(source), label="Violence", confidence=float(confidence), violence_probability=float(violence_prob)))
                db.commit()
        except Exception as e:
            print(f"[WORKER] Failed to save alert to DB: {e}")

        try:
            send_violence_email(
                subject="ALERTE LIVE: Violence detectee",
                body=(
                    "Une alerte 'Violence' a ete detectee sur le live stream.\n\n"
                    f"- Source: live\n"
                    f"- Camera/source: {source}\n"
                    f"- Confidence: {float(confidence):.3f}\n"
                    f"- Violence probability: {float(violence_prob):.3f}\n"
                    f"- Date (UTC): {datetime.utcnow().isoformat()}Z\n"
                    f"- Cooldown: {int(ALERT_COOLDOWN_S)}s\n"
                ),
            )
        except Exception as e:
            print(f"[WORKER] Failed to send email: {e}")


WORKER = StreamWorker()


def register_events(ns):
    @sio.event(namespace=ns)
    async def connect(sid, environ, auth):
        global EVENT_LOOP
        if EVENT_LOOP is None:
            EVENT_LOOP = asyncio.get_running_loop()
        await sio.emit("frame", {"label": "connected", "confidence": 0.0, "ts": datetime.utcnow().isoformat() + "Z"}, to=sid, namespace=ns)

    @sio.event(namespace=ns)
    async def start(sid, data):
        source = str((data or {}).get("source", "0"))
        threshold = float((data or {}).get("threshold", 0.7))
        WORKER.start(sid=sid, source=source, threshold=threshold, namespace=ns)
        await sio.emit("status", {"running": True}, to=sid, namespace=ns)

    @sio.event(namespace=ns)
    async def stop(sid):
        WORKER.stop(sid=sid)
        await sio.emit("status", {"running": False}, to=sid, namespace=ns)

    @sio.event(namespace=ns)
    async def disconnect(sid):
        WORKER.stop(sid=sid)


register_events("/")
register_events("/stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(sio_app, host="0.0.0.0", port=8001, log_level="info")

