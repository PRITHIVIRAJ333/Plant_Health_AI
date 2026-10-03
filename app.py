from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model


app = Flask(__name__)


# Model paths
MODEL_PATH = "model/plant_health_model.keras"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"


# Load model
model = None
scaler = None
encoder = None


if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

if os.path.exists(ENCODER_PATH):
    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:
            return jsonify({
                "success": False,
                "message": "Model not found. Please train the model first."
            })

        data = request.get_json()

        soil_moisture = float(data["soil_moisture"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        leaf_moisture = float(data["leaf_moisture"])
        sunlight = float(data["sunlight"])
        soil_ph = float(data["soil_ph"])


        # Create input array
        input_data = np.array([[
            soil_moisture,
            temperature,
            humidity,
            leaf_moisture,
            sunlight,
            soil_ph
        ]])


        # Scale input
        input_scaled = scaler.transform(input_data)


        # Prediction
        prediction = model.predict(input_scaled, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = float(
            np.max(prediction) * 100
        )


        # Convert prediction to class name
        if encoder is not None:
            result = encoder.inverse_transform(
                [predicted_index]
            )[0]
        else:
            classes = [
                "Healthy",
                "Needs_Attention",
                "Unhealthy"
            ]

            result = classes[predicted_index]


        # Explanation
        if result == "Healthy":

            message = (
                "The plant conditions look healthy. "
                "Continue regular watering, sunlight "
                "and soil monitoring."
            )

        elif result == "Needs_Attention":

            message = (
                "The plant may need some attention. "
                "Check moisture, temperature, sunlight "
                "and soil pH regularly."
            )

        else:

            message = (
                "The plant conditions appear unhealthy. "
                "Check watering, temperature, sunlight "
                "and soil condition immediately."
            )


        return jsonify({
            "success": True,
            "result": result,
            "confidence": round(confidence, 2),
            "message": message
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )