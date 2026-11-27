
# MoringaCare AI Backend - Stable Grad-CAM Only Version

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from io import BytesIO
import uuid

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
XAI_DIR = STATIC_DIR / "xai"
MODEL_PATH = BASE_DIR / "models" / "Hybrid_CNN_Transformer_Final.keras"

CLASS_NAMES = [
    "Bacterial_Spot_Leaf",
    "Healthy_Leaf",
    "Yellow_Leaf",
]

DISPLAY_NAMES = {
    "Bacterial_Spot_Leaf": "Bacterial Spot Leaf",
    "Healthy_Leaf": "Healthy Leaf",
    "Yellow_Leaf": "Yellow Leaf",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_SIZE = (224, 224)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
XAI_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

# ============================================================
# 2. FASTAPI APPLICATION AND CORS
# ============================================================

app = FastAPI(
    title="MoringaCare AI API",
    description="Hybrid CNN-Transformer based Moringa Leaf Disease Detection",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://moringacare-ai.vercel.app",
        "http://localhost:8443",
        "http://127.0.0.1:8443",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# ============================================================
# 3. TENSORFLOW CPU CONFIGURATION AND MODEL LOADING
# ============================================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

if tuple(model.input_shape) != (None, 224, 224, 3):
    raise RuntimeError(f"Unexpected model input shape: {model.input_shape}")

if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError("Model output count does not match CLASS_NAMES.")

try:
    grad_layer = model.get_layer("reshape")
except ValueError as exc:
    raise RuntimeError(
        "Grad-CAM layer 'reshape' was not found in the trained model."
    ) from exc

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[grad_layer.input, model.output],
)

print("MoringaCare AI model loaded successfully!")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Grad-CAM initialized in CPU mode.")

# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )
            image_array = np.asarray(image, dtype=np.float32)

        return tf.convert_to_tensor(
            image_array[None, ...],
            dtype=tf.float32,
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload JPG, PNG, or WEBP.",
        ) from exc

# ============================================================
# 5. SAVE EXPLANATION IMAGES
# ============================================================

def save_rgb_image(array: np.ndarray, filename: str) -> str:
    path = XAI_DIR / filename

    Image.fromarray(
        np.asarray(array, dtype=np.uint8)
    ).save(path, format="PNG")

    if not path.is_file():
        raise RuntimeError("Could not save Grad-CAM image.")

    return f"/static/xai/{filename}"

# ============================================================
# 6. DYNAMIC GRAD-CAM
# ============================================================

def generate_gradcam(
    image_tensor: tf.Tensor,
    class_index: int,
) -> np.ndarray:

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            image_tensor,
            training=False,
        )
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are unavailable.")

    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    heatmap = tf.reduce_sum(
        feature_maps[0]
        * channel_weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def save_gradcam_images(
    image_tensor: tf.Tensor,
    class_index: int,
) -> dict:

    heatmap = generate_gradcam(
        image_tensor,
        class_index,
    )

    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC,
    )
    heatmap = np.clip(heatmap, 0, 1)

    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap(
            np.uint8(heatmap * 255),
            cv2.COLORMAP_JET,
        ),
        cv2.COLOR_BGR2RGB,
    )

    original_rgb = np.clip(
        image_tensor[0].numpy(),
        0,
        255,
    ).astype(np.uint8)

    overlay_rgb = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    prediction_id = uuid.uuid4().hex

    return {
        "gradcam_heatmap": save_rgb_image(
            heatmap_rgb,
            f"{prediction_id}_gradcam_heatmap.png",
        ),
        "gradcam_overlay": save_rgb_image(
            overlay_rgb,
            f"{prediction_id}_gradcam_overlay.png",
        ),
    }

# ============================================================
# 7. BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to MoringaCare AI API",
        "status": "Backend is running",
        "version": "3.2.0",
        "xai": "Dynamic Grad-CAM only",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_enabled": grad_model is not None,
        "gradcam_mode": "dynamic_unique_urls",
        "runtime": "CPU",
        "xai_images_available": len(list(XAI_DIR.glob("*.png"))),
    }

# ============================================================
# 8. PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            if uploaded_image.format not in ("JPEG", "PNG", "WEBP"):
                raise HTTPException(
                    status_code=400,
                    detail="Please upload JPG, PNG, or WEBP.",
                )
            uploaded_image.verify()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        ) from exc

    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    all_predictions = {
        CLASS_NAMES[index]: round(
            float(predictions[0][index]) * 100,
            2,
        )
        for index in range(len(CLASS_NAMES))
    }

    try:
        explanations = save_gradcam_images(
            processed_image,
            predicted_index,
        )

    except Exception as exc:
        print("Grad-CAM generation error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail="Prediction completed, but Grad-CAM generation failed. Please try another image.",
        ) from exc

    return {
        "success": True,
        "predicted_class": predicted_class,
        "disease_name": DISPLAY_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "all_predictions": all_predictions,
        "explanations": explanations,
    }

# ============================================================
# 9. STARTUP INFORMATION
# ============================================================

print("XAI directory:", XAI_DIR)
print("Available Grad-CAM images:", len(list(XAI_DIR.glob("*.png"))))
print("MoringaCare AI API initialization completed!")
