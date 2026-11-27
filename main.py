
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
    "Yellow_Leaf"
]

DISPLAY_NAMES = {
    "Bacterial_Spot_Leaf": "Bacterial Spot Leaf",
    "Healthy_Leaf": "Healthy Leaf",
    "Yellow_Leaf": "Yellow Leaf"
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_SIZE = (224, 224)

for directory in (STATIC_DIR, XAI_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MoringaCare AI API",
    description="Hybrid CNN-Transformer based Moringa Leaf Disease Detection",
    version="2.1.0"
)

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

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if tuple(model.input_shape) != (None, 224, 224, 3):
    raise RuntimeError(
        f"Unexpected model input shape: {model.input_shape}"
    )

if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError(
        "Model output count does not match CLASS_NAMES."
    )

# Build Grad-CAM model using the actual saved model architecture.
grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("reshape").input,
        model.output
    ]
)

print("MoringaCare AI model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
print("Classes:", CLASS_NAMES)
print("Dynamic Grad-CAM model initialized!")


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes: bytes) -> tf.Tensor:

    try:
        with Image.open(BytesIO(image_bytes)) as image:

            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

        # Keep 0-255 pixel scale to match existing model pipeline.
        image_tensor = tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )

        return tf.expand_dims(
            image_tensor,
            axis=0
        )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid image. Please upload a valid JPG, PNG, or WEBP image."
        ) from exc


# ============================================================
# 5. DYNAMIC GRAD-CAM GENERATION
# ============================================================

def generate_gradcam(image_tensor: tf.Tensor, class_index: int):

    with tf.GradientTape() as tape:

        # Pass tensor directly, not [image_tensor].
        feature_maps, predictions = grad_model(
            image_tensor,
            training=False
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable."
        )

    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    heatmap = tf.reduce_sum(
        feature_maps[0] *
        channel_weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) +
        tf.keras.backend.epsilon()
    )

    result = heatmap.numpy()

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Grad-CAM produced non-finite values."
        )

    return result


# ============================================================
# 6. SAVE UNIQUE GRAD-CAM IMAGES
# ============================================================

def save_gradcam_images(image_tensor: tf.Tensor, class_index: int):

    heatmap = generate_gradcam(
        image_tensor,
        class_index
    )

    heatmap = cv2.resize(
        heatmap.astype(np.float32),
        IMAGE_SIZE,
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    original_rgb = np.clip(
        image_tensor[0].numpy(),
        0,
        255
    ).astype(np.uint8)

    overlay_rgb = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_rgb,
        0.40,
        0
    )

    # Generate a unique ID for every prediction.
    prediction_id = uuid.uuid4().hex

    heatmap_filename = f"{prediction_id}_heatmap.png"
    overlay_filename = f"{prediction_id}_overlay.png"

    heatmap_path = XAI_DIR / heatmap_filename
    overlay_path = XAI_DIR / overlay_filename

    Image.fromarray(heatmap_rgb).save(
        heatmap_path,
        format="PNG"
    )

    Image.fromarray(overlay_rgb).save(
        overlay_path,
        format="PNG"
    )

    if not heatmap_path.is_file() or not overlay_path.is_file():
        raise RuntimeError(
            "Failed to save Grad-CAM images."
        )

    return {
        "gradcam_heatmap": f"/static/xai/{heatmap_filename}",
        "gradcam_overlay": f"/static/xai/{overlay_filename}"
    }


# ============================================================
# 7. EXISTING LIME AND SHAP IMAGES
# ============================================================

def get_xai_urls():

    base = "/static/xai/"

    return {
        "lime": base + "lime_explanation.png",
        "shap": base + "shap_explanation.png",
        "shap_heatmap": base + "shap_explanation.png",
        "shap_visualization": base + "shap_explanation.png"
    }


# ============================================================
# 8. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MoringaCare AI API",
        "status": "Backend is running",
        "version": "2.1.0"
    }


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_enabled": grad_model is not None,
        "gradcam_mode": "dynamic_unique_urls",
        "xai_images_available": len(
            list(XAI_DIR.glob("*.png"))
        )
    }


# ============================================================
# 10. PREDICTION API
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > MAX_UPLOAD_BYTES:

        raise HTTPException(
            status_code=413,
            detail="Image size must be below 10 MB."
        )

    # Validate actual image format.
    try:

        with Image.open(BytesIO(image_bytes)) as uploaded_image:

            if uploaded_image.format not in (
                "JPEG",
                "PNG",
                "WEBP"
            ):

                raise HTTPException(
                    status_code=400,
                    detail="Please upload a JPG, PNG, or WEBP image."
                )

            uploaded_image.verify()

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        ) from exc

    # Preprocess uploaded image.
    processed_image = preprocess_image(
        image_bytes
    )

    # Model prediction.
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

    # Class probabilities.
    all_predictions = {
        CLASS_NAMES[index]: round(
            float(predictions[0][index]) * 100,
            2
        )
        for index in range(len(CLASS_NAMES))
    }

    # Generate NEW Grad-CAM images for this uploaded image.
    try:

        dynamic_gradcam_urls = save_gradcam_images(
            processed_image,
            predicted_index
        )

    except Exception as exc:

        print(
            "Grad-CAM generation error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction completed, but Grad-CAM generation failed."
        ) from exc

    # Keep existing LIME and SHAP sample images.
    explanations = get_xai_urls()

    # Replace Grad-CAM with new unique image URLs.
    explanations.update(
        dynamic_gradcam_urls
    )

    return {
        "success": True,
        "predicted_class": predicted_class,
        "disease_name": DISPLAY_NAMES[predicted_class],
        "confidence": round(confidence * 100, 2),
        "all_predictions": all_predictions,
        "explanations": explanations
    }


# ============================================================
# 11. SERVER INFORMATION
# ============================================================

print("XAI directory:", XAI_DIR)

print(
    "Available XAI images:",
    len(list(XAI_DIR.glob("*.png")))
)

print("MoringaCare AI API initialization completed!")
