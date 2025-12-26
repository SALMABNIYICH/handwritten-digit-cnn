from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

# -------------------
# Flask Configuration
# -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
SAMPLES_FOLDER = os.path.join("static/mnist_samples")

# -------------------
# Generate MNIST samples (once)
# -------------------
def generate_mnist_samples(n_images=6):
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)
    if len(os.listdir(SAMPLES_FOLDER)) == 0:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()

        for i in range(n_images):
            img = x_train[i]
            path = os.path.join(SAMPLES_FOLDER, f"{i}_{y_train[i]}.png")
            plt.imsave(path, img, cmap="gray")

        print(f"{n_images} MNIST images generated in {SAMPLES_FOLDER}")

generate_mnist_samples()

# -------------------
# Load trained CNN model
# -------------------
model = keras.models.load_model("mnist_cnn_model.h5")

# -------------------
# Utility functions
# -------------------
def prepare_image(img):
    if np.mean(img) > 127:
        img = 255 - img

    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def get_sample_images(n_images=6):
    os.makedirs(SAMPLES_FOLDER, exist_ok=True)
    files = [
        os.path.join(SAMPLES_FOLDER, f)
        for f in os.listdir(SAMPLES_FOLDER)
        if f.endswith(".png")
    ]
    return files[:n_images]

# -------------------
# Routes
# -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/explain")
def explain():
    images = get_sample_images()
    accuracy = round(float(np.load("accuracy.npy")) * 100, 2)
    return render_template("explain.html", images=images, accuracy=accuracy)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    file_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = prepare_image(img)
            preds = model.predict(img)[0]

            prediction = int(np.argmax(preds))
            confidence = round(float(np.max(preds)) * 100, 2)

        else:
            data = request.get_json()
            if data and "imageBase64" in data:
                image_data = data["imageBase64"].split(",")[1]
                img_bytes = base64.b64decode(image_data)

                pil_img = Image.open(BytesIO(img_bytes)).convert("L")
                img_np = prepare_image(np.array(pil_img))
                preds = model.predict(img_np)[0]

                prediction = int(np.argmax(preds))
                confidence = round(float(np.max(preds)) * 100, 2)

            return jsonify({
                "prediction": prediction,
                "confidence": confidence
            })

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        file_path=file_path
    )

# -------------------
# STATISTICS PAGE
# -------------------
@app.route("/statistics")
def statistics():
    metrics = np.load("metrics.npy", allow_pickle=True).item()
    history_file = "history.npy"

    # ---- Per-digit metrics ----
    classification = metrics["classification_report"]
    labels = [str(i) for i in range(10)]

    classification_data = {
        "labels": labels,
        "precision": [classification[l]["precision"] for l in labels],
        "recall": [classification[l]["recall"] for l in labels],
        "f1score": [classification[l]["f1-score"] for l in labels],
    }

    # ---- Confusion matrix ----
    confusion_matrix = metrics["confusion_matrix"].tolist()

    # ---- Training history ----
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        history_data = {
            "epochs": list(range(1, len(history["accuracy"]) + 1)),
            "accuracy": history["accuracy"],
            "val_accuracy": history.get("val_accuracy", history["accuracy"]),
            "loss": history["loss"],
            "val_loss": history.get("val_loss", history["loss"]),
        }
    else:
        acc = metrics["accuracy"]
        loss = metrics["loss"]
        history_data = {
            "epochs": [1, 2, 3, 4, 5],
            "accuracy": [acc] * 5,
            "val_accuracy": [acc] * 5,
            "loss": [loss] * 5,
            "val_loss": [loss] * 5,
        }

    return render_template(
        "statistics.html",
        classification_data=classification_data,
        confusion_matrix=confusion_matrix,
        accuracy=round(metrics["accuracy"] * 100, 2),
        loss=round(metrics["loss"], 4),
        history_data=history_data
    )

# -------------------
# Run server
# -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)






