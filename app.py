from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("model.h5")


@app.route("/")
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["image"]

        # Check if a file was uploaded
        if file:
            img = Image.open(file)
            img = img.resize((150, 150))  # Resize to match the model's input size
            img = np.array(img) / 255.0  # Normalize pixel values
            img = img.reshape(1, 150, 150, 3)  # Reshape for model input

            # Perform the prediction
            prediction = model.predict(img)

            # Determine the result based on the model's output
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
        else:
            result = "Please upload an image."

        return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
