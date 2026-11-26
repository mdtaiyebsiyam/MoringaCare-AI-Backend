
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# =====================================
# APPLICATION CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
XAI_DIR = STATIC_DIR / "xai"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "Hybrid_CNN_Transformer_Final.keras"
)

if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

if not XAI_DIR.is_dir():
    raise RuntimeError(f"XAI directory not found: {XAI_DIR}")

if not MODEL_PATH.is_file():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")


app = FastAPI(
    title="MoringaCare AI API",
    description="Hybrid CNN-Transformer based Moringa Leaf Disease Detection",
    version="1.0.0"
)


# =====================================
# CORS CONFIGURATION
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8443",
        "http://127.0.0.1:8443",
        "https://moringacare-ai.vercel.app"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# =====================================
# STATIC FILE CONFIGURATION
# =====================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# =====================================
# MODEL CONFIGURATION
# =====================================

CLASS_NAMES = [
    "Bacterial_Spot_Leaf",
    "Healthy_Leaf",
    "Yellow_Leaf"
]

DISPLAY_NAMES = {
    "Bacterial_Spot_Leaf": "Bacterial Spot Leaf",
    "Healthy_Leaf": "Healthy Leaf",
    "Yellow_Leaf": "Yellow Leaf"
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if model.input_shape != (None, 224, 224, 3):
    raise RuntimeError(
        f"Unexpected model input shape: {model.input_shape}"
    )

if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError(
        "Model output count does not match the number of classes."
    )

print("MoringaCare AI model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Classes:", CLASS_NAMES)


# =====================================
# XAI IMAGE URLS
# =====================================

def get_xai_urls():

    base_url = "/static/xai/"

    return {
        "gradcam_heatmap": base_url + "gradcam_heatmap.png",
        "gradcam_overlay": base_url + "gradcam_overlay.png",
        "lime": base_url + "lime_explanation.png",
        "shap": base_url + "shap_explanation.png",
        "shap_heatmap": base_url + "shap_explanation.png",
        "shap_visualization": base_url + "shap_explanation.png"
    }


# =====================================
# IMAGE PREPROCESSING
# =====================================

def preprocess_image(image_bytes):

    try:
        with Image.open(BytesIO(image_bytes)) as image:

            image = image.convert("RGB")

            image_array = np.asarray(
                image,
                dtype=np.uint8
            )

        # Match the training pipeline:
        # RGB image -> 224x224 resize -> float32
        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        image_tensor = tf.image.resize(
            image_tensor,
            [224, 224],
            method="bilinear"
        )

        image_tensor = tf.expand_dims(
            image_tensor,
            axis=0
        )

        return image_tensor

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid JPG, PNG, or WEBP image."
        )


# =====================================
# API ROUTES
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MoringaCare AI API",
        "status": "Backend is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "xai_images_available": len(
            list(XAI_DIR.glob("*.png"))
        )
    }


# =====================================
# LEAF DISEASE PREDICTION
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    # Validate empty image
    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    # Validate file size (10 MB maximum)
    if len(image_bytes) > 10 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    # Validate actual image format
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:

            if uploaded_image.format not in [
                "JPEG",
                "PNG",
                "WEBP"
            ]:

                raise HTTPException(
                    status_code=400,
                    detail="Please upload a JPG, PNG, or WEBP image."
                )

            uploaded_image.verify()

    except HTTPException:
        raise

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    # Preprocess image
    processed_image = preprocess_image(image_bytes)

    # Model prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    )

    # Class probabilities in percentage
    all_predictions = {
        CLASS_NAMES[i]: round(
            float(predictions[0][i]) * 100,
            2
        )
        for i in range(len(CLASS_NAMES))
    }

    # Explanation image URLs
    explanations = get_xai_urls()

    return {
        "success": True,
        "predicted_class": predicted_class,
        "disease_name": DISPLAY_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "all_predictions": all_predictions,
        "explanations": explanations
    }


# =====================================
# SERVER INFORMATION
# =====================================

print("XAI directory:", XAI_DIR)
print(
    "Available XAI images:",
    len(list(XAI_DIR.glob("*.png")))
)
