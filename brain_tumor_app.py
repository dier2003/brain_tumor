"""
====================================================================
 BRAIN TUMOR MRI CLASSIFICATION — PRESENTATION-READY GRADIO APP
====================================================================
Run this file with:
    pip install tensorflow gradio pillow numpy plotly
    python brain_tumor_app.py

It will launch a polished, presentation-style web interface for the
CNN model trained in brain_tumor.ipynb.

Before running, make sure MODEL_PATH below points to your saved
model file (brain_tumor_classifier_model.keras).
====================================================================
"""

import numpy as np
import tensorflow as tf
import gradio as gr
import plotly.graph_objects as go
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "brain_tumor_classifier_model.keras"  # update path if needed
IMG_SIZE = (128, 128)

# Must match the order used during training (alphabetical folder order)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": "A tumor that starts in the glial cells of the brain or spine.",
    "Meningioma": "A tumor arising from the meninges, the membranes covering the brain.",
    "No Tumor": "No visible tumor detected in the scanned MRI slice.",
    "Pituitary": "A tumor located in the pituitary gland at the base of the brain.",
}

CLASS_ICONS = {
    "Glioma": "🔴",
    "Meningioma": "🟠",
    "No Tumor": "🟢",
    "Pituitary": "🔵",
}

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded. Input shape:", model.input_shape)


# ============================================================
# PREDICTION LOGIC
# ============================================================

def make_probability_chart(probabilities: dict) -> go.Figure:
    """Build a clean horizontal bar chart of class probabilities."""
    labels = list(probabilities.keys())
    values = [probabilities[k] * 100 for k in labels]
    colors = ["#ef4444", "#f97316", "#22c55e", "#3b82f6"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors[: len(labels)]),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Confidence (%)"),
        yaxis=dict(autorange="reversed"),
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def predict_tumor(img: Image.Image):
    if img is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)")
        return (
            "### ⚠️ Please upload a brain MRI image to begin.",
            empty_fig,
            "Waiting for image...",
        )

    # Preprocess
    processed = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(processed).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    probabilities = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
    chart = make_probability_chart(probabilities)

    icon = CLASS_ICONS.get(predicted_class, "🧠")
    description = CLASS_INFO.get(predicted_class, "")

    result_md = f"""
### {icon} Predicted Class: **{predicted_class}**

**Confidence:** {confidence:.2f}%

> {description}
"""

    if confidence >= 90:
        status = "🟢 **High confidence** — the model is very sure about this result."
    elif confidence >= 70:
        status = "🟡 **Moderate confidence** — result is likely correct but review is advised."
    else:
        status = "🔴 **Low confidence** — result is uncertain, consider a clearer image."

    return result_md, chart, status


def clear_all():
    empty_fig = go.Figure()
    empty_fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)")
    return None, "### Waiting for image", empty_fig, "Prediction status will appear here."


# ============================================================
# CUSTOM CSS — PRESENTATION LOOK
# ============================================================

CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

#hero {
    text-align: center;
    padding: 40px 20px;
    border-radius: 22px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
    color: white;
    box-shadow: 0 10px 30px rgba(30, 58, 138, 0.35);
}
#hero h1 {
    font-size: 34px;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
#hero p {
    font-size: 16px;
    opacity: 0.9;
    margin: 2px 0;
}

.card {
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
}

.disclaimer {
    padding: 18px 22px;
    border-radius: 16px;
    margin-top: 22px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    font-size: 14px;
}

.footer {
    text-align: center;
    padding: 18px;
    color: #64748b;
    font-size: 13px;
}

button.primary {
    border-radius: 12px !important;
}
"""

# ============================================================
# BUILD THE UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, title="Brain Tumor Classifier") as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1>🧠 Brain Tumor MRI Classification</h1>
            <p>AI-powered detection using a Convolutional Neural Network</p>
            <p>TensorFlow • Deep Learning • Gradio</p>
        </div>
        """
    )

    with gr.Tabs():

        # ---------------- CLASSIFY TAB ----------------
        with gr.TabItem("🔍 Classify"):

            with gr.Row():
                with gr.Column(scale=1, elem_classes="card"):
                    gr.Markdown("## 📤 Upload MRI Scan")
                    image_input = gr.Image(type="pil", label="Brain MRI Image", height=340)

                    with gr.Row():
                        analyze_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
                        clear_btn = gr.Button("🔄 Clear", size="lg")

                with gr.Column(scale=1, elem_classes="card"):
                    gr.Markdown("## 📊 Result")
                    result_output = gr.Markdown("### Waiting for image")
                    status_output = gr.Markdown("Prediction status will appear here.")
                    gr.Markdown("### Class Probabilities")
                    prob_chart = gr.Plot()

            analyze_btn.click(
                fn=predict_tumor,
                inputs=image_input,
                outputs=[result_output, prob_chart, status_output],
            )
            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[image_input, result_output, prob_chart, status_output],
            )

        # ---------------- ABOUT TAB ----------------
        with gr.TabItem("ℹ️ About the Model"):
            gr.Markdown(
                """
## 🔬 How It Works

**1. Upload** — Provide a brain MRI scan (JPG/PNG).
**2. Preprocessing** — Image is resized to 128×128 and pixel values normalized to [0, 1].
**3. CNN Inference** — A trained Convolutional Neural Network analyzes the scan.
**4. Classification** — The model predicts one of four categories:

| Class | Description |
|---|---|
| 🔴 Glioma | Tumor in the glial cells of the brain or spine |
| 🟠 Meningioma | Tumor arising from the meninges |
| 🟢 No Tumor | No visible tumor detected |
| 🔵 Pituitary | Tumor in the pituitary gland |

**5. Result** — Predicted class and confidence score are displayed with a probability breakdown.
                """
            )

    gr.HTML(
        """
        <div class="disclaimer">
        <b>⚠️ Important Notice</b>
        <p>This application is an educational/research demonstration of a deep learning
        classification model. It is <b>not</b> a medical diagnostic tool. Always consult
        a qualified healthcare professional for medical interpretation of MRI scans.</p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="footer">
            🧠 Brain Tumor MRI Classification System — Built with TensorFlow, CNN &amp; Gradio
        </div>
        """
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
