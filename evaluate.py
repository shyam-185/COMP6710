from PIL import Image
import pandas as pd
import numpy as np
from haversine import haversine
import matplotlib.pyplot as plt
from regression_model import predict_coordinates, adjust_to_boundary
from classifier_model import classify

##############################################################################################
#                                                                                            #
#                                  Evalute Model                                             #
#                                                                                            #
##############################################################################################

# Load test metadata
test_file = "data/test_metadata.csv"
try:
    metadata = pd.read_csv(test_file)
    print(f"Loaded test metadata with {len(metadata)} rows.")
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit(1)

# Load and process image data
def load_and_process_image(image):
    try:
        img = Image.open(image).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image {image}: {e}")
        return None

################################### Run Models ###############################################

results = []

# Iterate through metadata and process each row
for index, row in metadata.iterrows():
    image = row["image"]
    actual_lat, actual_lon = row["latitude"], row["longitude"]
    processed_img = load_and_process_image(image)
    if processed_img is None:
        print(f"Skipping invalid image: {image}")
        continue

    # Predict country
    country_scores = classify(processed_img)
    predicted_country, confidence = max(country_scores.items(), key=lambda item: item[1])
    confidence *= 100


    # Predict coordinates
    p_lat, p_lon, input_features = predict_coordinates(image, predicted_country)

    # Adjust coordinates
    a_lat, a_lon = adjust_to_boundary(p_lat, p_lon, predicted_country, input_features)

    # Haversine distance calculations for metrics
    haversine_distance = haversine((actual_lat, actual_lon), (p_lat, p_lon))
    haversine_distance_adjusted = haversine((actual_lat, actual_lon), (a_lat, a_lon))

    results.append({
        "Image": image,
        "Actual Latitude": actual_lat,
        "Actual Longitude": actual_lon,
        "Predicted Latitude": p_lat,
        "Predicted Longitude": p_lon,
        "Adjusted Latitude": a_lat,
        "Adjusted Longitude": a_lon,
        "Haversine Distance Original": haversine_distance,
        "Haversine Distance Adjusted": haversine_distance_adjusted,
        "Predicted Country": predicted_country,
        "Country Confidence": confidence
    })

results_df = pd.DataFrame(results)

################################### Graphs and Tables ###########################################

# Calculate metrics
mean_original = results_df['Haversine Distance Original'].mean()
mean_adjusted = results_df['Haversine Distance Adjusted'].mean()
closest_original = results_df['Haversine Distance Original'].min()
closest_adjusted = results_df['Haversine Distance Adjusted'].min()
farthest_original = results_df['Haversine Distance Original'].max()
farthest_adjusted = results_df['Haversine Distance Adjusted'].max()

catgories = [
    "<=50 km", 
    "<=500 km", 
    "<=1000 km", 
    "<=5000 km", 
    ">=5000 km"
    ]
distance_bins = [0, 
                 50, 
                 500, 
                 1000, 
                 5000, 
                 float('inf')
                 ]

# Original
results_df['Distance Category Original'] = pd.cut(
    results_df['Haversine Distance Original'],
    bins=distance_bins,
    labels=catgories,
    include_lowest=True
)

# Adjusted
results_df['Distance Category Adjusted'] = pd.cut(
    results_df['Haversine Distance Adjusted'],
    bins=distance_bins,
    labels=catgories,
    include_lowest=True
)

results_df.to_csv("evaluation_results.csv", index=False)
print("Results saved to evaluation_results.csv")

# I used chatgpt here to help me produce proper graphs and tables
# Tables
distance_table = pd.DataFrame({
    "Distance Category": catgories,
    "Original Predictions": results_df['Distance Category Original'].value_counts().reindex(catgories, fill_value=0),
    "Adjusted Predictions": results_df['Distance Category Adjusted'].value_counts().reindex(catgories, fill_value=0),
})

summary_table = pd.DataFrame({
    "Metric": ["Mean Distance (Original)", "Mean Distance (Adjusted)",
               "Closest Point (Original)", "Closest Point (Adjusted)",
               "Furthest Point (Original)", "Furthest Point (Adjusted)"],
    "Value": [mean_original, mean_adjusted,
              closest_original, closest_adjusted,
              farthest_original, farthest_adjusted]
})

distance_table.to_csv("distance_table.csv", index=False)
summary_table.to_csv("summary_table.csv", index=False)

# Scatter plots
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Actual Longitude'], results_df['Actual Latitude'], label="Actual Points", alpha=0.7)
plt.scatter(results_df['Predicted Longitude'], results_df['Predicted Latitude'], label="Original Predicted Points", alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Actual v Original Points")
plt.legend()
plt.savefig("scatter_actual_vs_original.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(results_df['Actual Longitude'], results_df['Actual Latitude'], label="Actual Points", alpha=0.7)
plt.scatter(results_df['Adjusted Longitude'], results_df['Adjusted Latitude'], label="Adjusted Predicted Points", alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Actual v Adjusted Points")
plt.legend()
plt.savefig("scatter_actual_vs_adjusted.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(results_df['Predicted Longitude'], results_df['Predicted Latitude'], label="Original Predicted Points", alpha=0.7)
plt.scatter(results_df['Adjusted Longitude'], results_df['Adjusted Latitude'], label="Adjusted Predicted Points", alpha=0.7)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Original v Adjusted Points")
plt.legend()
plt.savefig("scatter_original_vs_adjusted.png")
plt.close()
