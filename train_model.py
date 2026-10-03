import csv
import json
import random
import math

random.seed(42)


def relu(x):
    return max(0.0, x)


def relu_derivative(x):
    if x > 0:
        return 1.0
    return 0.0


def softmax(values):
    maximum = max(values)

    exp_values = [
        math.exp(x - maximum)
        for x in values
    ]

    total = sum(exp_values)

    return [
        x / total
        for x in exp_values
    ]


# ==============================
# LOAD DATASET
# ==============================

X = []
y = []

with open("dataset.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        features = [
            float(row["Soil_Moisture"]),
            float(row["Temperature"]),
            float(row["Humidity"]),
            float(row["Leaf_Moisture"]),
            float(row["Sunlight"]),
            float(row["Soil_pH"])
        ]

        X.append(features)

        label = row["Plant_Health"]

        if label == "Healthy":
            y.append(0)

        elif label == "Needs_Attention":
            y.append(1)

        else:
            y.append(2)


print("====================================")
print("Dataset loaded successfully!")
print("Total records:", len(X))
print("====================================")


# ==============================
# NORMALIZATION
# ==============================

feature_count = 6

means = []
stds = []

for j in range(feature_count):

    values = [
        X[i][j]
        for i in range(len(X))
    ]

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = math.sqrt(variance)

    if std == 0:
        std = 1

    means.append(mean)
    stds.append(std)


for i in range(len(X)):

    for j in range(feature_count):

        X[i][j] = (
            X[i][j] - means[j]
        ) / stds[j]


# ==============================
# SHUFFLE
# ==============================

combined = list(zip(X, y))

random.shuffle(combined)

X = [
    item[0]
    for item in combined
]

y = [
    item[1]
    for item in combined
]


# ==============================
# TRAIN / TEST SPLIT
# ==============================

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==============================
# NEURAL NETWORK
# ==============================

input_size = 6
hidden_size = 16
output_size = 3


W1 = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(hidden_size)
    ]
    for _ in range(input_size)
]

b1 = [
    0.0
    for _ in range(hidden_size)
]


W2 = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(output_size)
    ]
    for _ in range(hidden_size)
]

b2 = [
    0.0
    for _ in range(output_size)
]


# ==============================
# FORWARD PROPAGATION
# ==============================

def forward(inputs):

    hidden_raw = []

    for j in range(hidden_size):

        value = b1[j]

        for i in range(input_size):

            value += (
                inputs[i] *
                W1[i][j]
            )

        hidden_raw.append(value)


    hidden = [
        relu(value)
        for value in hidden_raw
    ]


    output_raw = []

    for k in range(output_size):

        value = b2[k]

        for j in range(hidden_size):

            value += (
                hidden[j] *
                W2[j][k]
            )

        output_raw.append(value)


    output = softmax(output_raw)

    return (
        hidden_raw,
        hidden,
        output_raw,
        output
    )


# ==============================
# TRAINING
# ==============================

epochs = 80
learning_rate = 0.01

print("")
print("Training neural network...")
print("")


for epoch in range(epochs):

    correct = 0
    total_loss = 0.0


    for inputs, target in zip(
        X_train,
        y_train
    ):

        (
            hidden_raw,
            hidden,
            output_raw,
            output
        ) = forward(inputs)


        loss = -math.log(
            max(
                output[target],
                1e-10
            )
        )

        total_loss += loss


        prediction = output.index(
            max(output)
        )


        if prediction == target:

            correct += 1


        # Output gradient

        output_gradient = output[:]

        output_gradient[target] -= 1


        old_W2 = [
            row[:]
            for row in W2
        ]


        # Update W2

        for j in range(hidden_size):

            for k in range(output_size):

                W2[j][k] -= (
                    learning_rate *
                    hidden[j] *
                    output_gradient[k]
                )


        # Update b2

        for k in range(output_size):

            b2[k] -= (
                learning_rate *
                output_gradient[k]
            )


        # Hidden gradient

        hidden_gradient = [
            0.0
            for _ in range(hidden_size)
        ]


        for j in range(hidden_size):

            value = 0.0

            for k in range(output_size):

                value += (
                    output_gradient[k] *
                    old_W2[j][k]
                )


            hidden_gradient[j] = (
                value *
                relu_derivative(
                    hidden_raw[j]
                )
            )


        # Update W1

        for i in range(input_size):

            for j in range(hidden_size):

                W1[i][j] -= (
                    learning_rate *
                    inputs[i] *
                    hidden_gradient[j]
                )


        # Update b1

        for j in range(hidden_size):

            b1[j] -= (
                learning_rate *
                hidden_gradient[j]
            )


    accuracy = (
        correct /
        len(X_train)
    ) * 100


    average_loss = (
        total_loss /
        len(X_train)
    )


    if (
        epoch == 0 or
        (epoch + 1) % 10 == 0
    ):

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f} "
            f"- Accuracy: {accuracy:.2f}%"
        )


# ==============================
# TESTING
# ==============================

correct_test = 0


for inputs, target in zip(
    X_test,
    y_test
):

    (
        hidden_raw,
        hidden,
        output_raw,
        output
    ) = forward(inputs)


    prediction = output.index(
        max(output)
    )


    if prediction == target:

        correct_test += 1


test_accuracy = (
    correct_test /
    len(X_test)
) * 100


print("")
print("====================================")
print(
    "Test Accuracy:",
    round(test_accuracy, 2),
    "%"
)
print("====================================")


# ==============================
# SAVE MODEL
# ==============================

model = {

    "input_size": input_size,

    "hidden_size": hidden_size,

    "output_size": output_size,

    "W1": W1,

    "b1": b1,

    "W2": W2,

    "b2": b2,

    "means": means,

    "stds": stds,

    "classes": [
        "Healthy",
        "Needs_Attention",
        "Unhealthy"
    ]
}


with open(
    "model.json",
    "w"
) as file:

    json.dump(
        model,
        file
    )


print("")
print("Model saved successfully!")
print("File: model.json")