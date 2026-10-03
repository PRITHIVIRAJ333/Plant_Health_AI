import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Load dataset
data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))

# Input columns
X = data[
    [
        "Soil_Moisture",
        "Temperature",
        "Humidity",
        "Leaf_Moisture",
        "Sunlight",
        "Soil_pH"
    ]
].values

# Output column
y = data["Plant_Health"].values

# Convert text labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Classes:", encoder.classes_)

# Convert labels to categorical
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Scale input values
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create ANN model
model = Sequential([
    Dense(32, activation="relu", input_shape=(6,)),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.1),

    Dense(8, activation="relu"),

    Dense(3, activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model
print("\nTraining model...")

history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)


# Test model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nModel training completed!")
print("Test Accuracy:", round(accuracy * 100, 2), "%")


# Create model folder
os.makedirs("model", exist_ok=True)


# Save model
model.save("model/plant_health_model.keras")


# Save scaler
import pickle

with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


# Save label encoder
with open("model/label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)


print("\nModel saved successfully!")
print("Location: model/plant_health_model.keras")
print("Scaler saved successfully!")
print("Label encoder saved successfully!")