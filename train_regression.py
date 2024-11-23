import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import json
from shapely.geometry import shape

##############################################################################################
#                                                                                            #
#                                     Model Training                                         #
#                                                                                            #
##############################################################################################

# Paths
metadata_file = "data/full_metadata.csv"
features_file = "data/streetclip_features.npy"
geojson_file = "data/world-administrative-boundaries.geojson"
output_model_file = "regression_model.pkl"

def load_geojson_centroids(geojson_file):
"""Load GeoJSON to extract centroids. This is technically just the representation point of a country.
This is needed in the calculation of the adjusted prediction, so we do not adjust to an edge of the country shape.

Params:
-------
    geojson_file (string):
        Geojson file that contains shapes and other data of countries.

"""
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)
    centroids = {}
    for feature in geojson_data["features"]:
        try:
            country_name = feature["properties"]["name"]
            geo_point = feature["properties"]["geo_point_2d"]
            centroids[country_name] = (geo_point["lat"], geo_point["lon"])
        except KeyError as e:
            print(f"Missing key in GeoJSON for {country_name}: {e}")
        except Exception as e:
            print(f"Error processing {country_name}: {e}")
    return centroids

centroids = load_geojson_centroids(geojson_file)

def correct_image_path(image_path):
"""Helper Function to correct paths, when running into frustrating errors to do with structure.

Params:
-------
    image_path (sting): 
        Relative or absoulte path to data file.
"""
    if not os.path.isfile(image_path):
        base_dir = "data/streetviews/"
        corrected_path = os.path.join(base_dir, '/'.join(image_path.split('/')[-2:]))
        if os.path.isfile(corrected_path):
            return corrected_path
        else:
            return None
    return image_path

# Load metadata
metadata = pd.read_csv(metadata_file)
metadata["image"] = metadata["image"].apply(correct_image_path)
metadata.dropna(subset=["image"], inplace=True)
print(f"Metadata after correcting paths: {len(metadata)} rows remain.")

# Check that feature file exists
if not os.path.exists(features_file):
    print("Feature file not found. Run 'feature_extractor.py' first.")
    exit(1)

features = np.load(features_file)
print(f"Feature File Dimensions: {features.shape}")

if len(metadata) != len(features):
    raise ValueError("Mismatch between metadata and feature file dimensions.")

# Assign Features to Metadata
metadata["clip_features"] = list(features)
print("Features successfully loaded and mapped to metadata.")

# Map centroids to metadata
if "country_code" not in metadata.columns:
    raise KeyError("The metadata file must include a 'country_code' column for mapping centroids.")

metadata["centroid"] = metadata["country_code"].map(
    lambda code: centroids.get(code, (0.0, 0.0))
)
metadata.dropna(subset=["centroid"], inplace=True)

# Prepare Features with Centroid
centroid_features = np.array(list(metadata["centroid"]))
reduced_features = np.vstack(metadata["clip_features"].values)[:, :4]
final_features = np.hstack([reduced_features, centroid_features])

if len(final_features) != len(metadata):
    print("Mismatch between metadata rows and final features.")
    exit(1)

# Train-Test Split
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)
print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples.")

X_train = np.hstack([
    np.stack(train_data["clip_features"].values),
    np.array(list(train_data["centroid"]))
])
y_train = train_data[["latitude", "longitude"]].values

# Train Regression Model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
print("Regression model training complete.")

# Save Model
with open(output_model_file, "wb") as f:
    pickle.dump(regressor, f)
print(f"Trained regression model saved to {output_model_file}.")
