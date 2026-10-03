from flask import Flask, render_template, request, jsonify
import json
import math
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.json"
)


with open(
    MODEL_PATH,
    "r"
) as file:

    model = json.load(file)


W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

means = model["means"]
stds = model["stds"]

classes = model["classes"]


def relu(x):

    return max(
        0.0,
        x
    )


def softmax(values):

    maximum = max(values)

    exp_values = [
        math.exp(
            value - maximum
        )
        for value in values
    ]

    total = sum(
        exp_values
    )

    return [
        value / total
        for value in exp_values
    ]


def predict(values):

    normalized = []

    for i in range(6):

        value = (
            values[i] -
            means[i]
        ) / stds[i]

        normalized.append(value)


    hidden = []

    for j in range(
        len(b1)
    ):

        value = b1[j]

        for i in range(6):

            value += (
                normalized[i] *
                W1[i][j]
            )

        hidden.append(
            relu(value)
        )


    output = []

    for k in range(3):

        value = b2[k]

        for j in range(
            len(hidden)
        ):

            value += (
                hidden[j] *
                W2[j][k]
            )

        output.append(value)


    probabilities = softmax(
        output
    )


    index = probabilities.index(
        max(probabilities)
    )


    result = classes[index]

    confidence = (
        probabilities[index] *
        100
    )


    return result, confidence


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def prediction():

    try:

        data = request.get_json()


        values = [

            float(
                data["soil_moisture"]
            ),

            float(
                data["temperature"]
            ),

            float(
                data["humidity"]
            ),

            float(
                data["leaf_moisture"]
            ),

            float(
                data["sunlight"]
            ),

            float(
                data["soil_ph"]
            )
        ]


        result, confidence = predict(
            values
        )


        if result == "Healthy":

            message = (
                "The plant conditions look "
                "healthy. Continue regular monitoring."
            )

        elif result == "Needs_Attention":

            message = (
                "The plant may need some attention. "
                "Check moisture, sunlight and soil pH."
            )

        else:

            message = (
                "The plant conditions appear unhealthy. "
                "Check the plant environment carefully."
            )


        return jsonify({

            "success": True,

            "result": result,

            "confidence": round(
                confidence,
                2
            ),

            "message": message
        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message": str(error)
        })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )