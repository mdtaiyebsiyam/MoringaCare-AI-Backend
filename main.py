
from pathlib import Path
from io import BytesIO
import uuid

import cv2
import numpy as np
import tensorflow as tf

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
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

LIME_NUM_SAMPLES = 100
LIME_BATCH_SIZE = 16

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
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8443",
        "http://127.0.0.1:8443",
        "https://moringacare-ai.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

if tuple(model.input_shape) != (None, 224, 224, 3):
    raise RuntimeError(f"Unexpected model input shape: {model.input_shape}")

if model.output_shape[-1] != len(CLASS_NAMES):
    raise RuntimeError("Model output count does not match CLASS_NAMES.")

# Keep the previously working Grad-CAM architecture.
grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("reshape").input,
        model.output,
    ],
)

print("MoringaCare AI model loaded successfully!")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Dynamic Grad-CAM and LIME initialized! SHAP is temporarily disabled.")


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
            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

        # Keep 0-255 pixel scale.
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


def predict_probabilities(images: np.ndarray) -> np.ndarray:
    batch = np.asarray(images, dtype=np.float32)

    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    probabilities = model.predict(
        batch,
        batch_size=LIME_BATCH_SIZE,
        verbose=0,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if probabilities.ndim != 2:
        raise RuntimeError("Unexpected model prediction shape.")

    return probabilities


def save_rgb_image(array: np.ndarray, filename: str) -> str:
    path = XAI_DIR / filename

    Image.fromarray(
        np.asarray(array, dtype=np.uint8)
    ).save(path, format="PNG")

    if not path.is_file():
        raise RuntimeError("Could not save explanation image.")

    return f"/static/xai/{filename}"


# ============================================================
# 5. DYNAMIC GRAD-CAM
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

    gradients = tape.gradient(
        class_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients unavailable.")

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
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
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
# 6. DYNAMIC LIME
# ============================================================

def generate_lime(
    image_tensor: tf.Tensor,
    class_index: int,
) -> dict:

    image_rgb = np.clip(
        image_tensor[0].numpy(),
        0,
        255,
    ).astype(np.uint8)

    explainer = LimeImageExplainer(
        random_state=42,
    )

    lime_input = image_rgb.astype(np.float32) / 255.0

    def lime_classifier(images):
        model_images = np.clip(
            np.asarray(images, dtype=np.float32) * 255.0,
            0,
            255,
        )

        return predict_probabilities(model_images)

    explanation = explainer.explain_instance(
        image=lime_input,
        classifier_fn=lime_classifier,
        labels=(class_index,),
        top_labels=None,
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES,
        batch_size=LIME_BATCH_SIZE,
    )

    explained_image, segments = explanation.get_image_and_mask(
        label=class_index,
        positive_only=False,
        num_features=8,
        hide_rest=False,
    )

    lime_visual = mark_boundaries(
        np.clip(explained_image, 0, 1),
        segments,
        color=(1, 1, 0),
        mode="thick",
    )

    lime_visual = np.uint8(
        np.clip(lime_visual, 0, 1) * 255
    )

    filename = f"{uuid.uuid4().hex}_lime.png"

    return {
        "lime": save_rgb_image(
            lime_visual,
            filename,
        ),
    }


# ============================================================
# 7. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MoringaCare AI API",
        "status": "Backend is running",
        "version": "3.1.0",
    }


# ============================================================
# 8. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_enabled": grad_model is not None,
        "lime_enabled": True,
        "shap_enabled": False,
        "xai_mode": "dynamic_gradcam_lime",
        "xai_images_available": len(
            list(XAI_DIR.glob("*.png"))
        ),
    }


# ============================================================
# 9. PREDICTION API
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

            if uploaded_image.format not in (
                "JPEG",
                "PNG",
                "WEBP",
            ):
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

    processed_image = preprocess_image(
        image_bytes,
    )

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    )

    all_predictions = {
        CLASS_NAMES[index]: round(
            float(predictions[0][index]) * 100,
            2,
        )
        for index in range(len(CLASS_NAMES))
    }

    # Generate all explanations for this uploaded image.
    try:

        explanations = {}

        explanations.update(
            save_gradcam_images(
                processed_image,
                predicted_index,
            )
        )

        explanations.update(
            generate_lime(
                processed_image,
                predicted_index,
            )
        )

    except Exception as exc:

        print(
            "XAI generation error:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction was made, but Grad-CAM/LIME generation failed. "
                "Check Render logs."
            ),
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
# 10. SERVER INFORMATION
# ============================================================

print("XAI directory:", XAI_DIR)

print(
    "Available XAI images:",
    len(list(XAI_DIR.glob("*.png"))),
)

print("MoringaCare AI API initialization completed!")
