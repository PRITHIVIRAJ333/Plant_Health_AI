import csv
import random

random.seed(42)

file_name = "dataset.csv"

rows = []

# HEALTHY - 334 rows
for i in range(334):

    soil_moisture = random.uniform(55, 80)
    temperature = random.uniform(22, 31)
    humidity = random.uniform(58, 80)
    leaf_moisture = random.uniform(50, 75)
    sunlight = random.uniform(55, 88)
    soil_ph = random.uniform(5.8, 7.2)

    rows.append([
        round(soil_moisture, 1),
        round(temperature, 1),
        round(humidity, 1),
        round(leaf_moisture, 1),
        round(sunlight, 1),
        round(soil_ph, 1),
        "Healthy"
    ])


# NEEDS ATTENTION - 333 rows
for i in range(333):

    soil_moisture = random.uniform(40, 60)
    temperature = random.uniform(28, 36)
    humidity = random.uniform(48, 68)
    leaf_moisture = random.uniform(35, 55)
    sunlight = random.uniform(65, 95)
    soil_ph = random.uniform(6.0, 7.8)

    rows.append([
        round(soil_moisture, 1),
        round(temperature, 1),
        round(humidity, 1),
        round(leaf_moisture, 1),
        round(sunlight, 1),
        round(soil_ph, 1),
        "Needs_Attention"
    ])


# UNHEALTHY - 333 rows
for i in range(333):

    soil_moisture = random.uniform(15, 42)
    temperature = random.uniform(35, 43)
    humidity = random.uniform(25, 48)
    leaf_moisture = random.uniform(12, 35)
    sunlight = random.uniform(85, 100)
    soil_ph = random.uniform(7.7, 8.8)

    rows.append([
        round(soil_moisture, 1),
        round(temperature, 1),
        round(humidity, 1),
        round(leaf_moisture, 1),
        round(sunlight, 1),
        round(soil_ph, 1),
        "Unhealthy"
    ])


random.shuffle(rows)

with open(file_name, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Soil_Moisture",
        "Temperature",
        "Humidity",
        "Leaf_Moisture",
        "Sunlight",
        "Soil_pH",
        "Plant_Health"
    ])

    writer.writerows(rows)


print("====================================")
print("Dataset created successfully!")
print("File:", file_name)
print("Total data rows:", len(rows))
print("Healthy:", 334)
print("Needs_Attention:", 333)
print("Unhealthy:", 333)
print("====================================")