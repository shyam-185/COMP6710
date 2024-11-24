import json
from shapely.geometry import Point, shape
from PIL import Image
from classifier_model import classify
from transformers import CLIPProcessor, CLIPModel
from shapely.ops import nearest_points
import torch
import pickle
import numpy as np

##############################################################################################
#                                                                                            #
#                                    Regression Model                                        #
#                                                                                            #
##############################################################################################


####################################### Load #################################################

# Load the StreetCLIP model and processor
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# Load the regression model
with open("regression_model.pkl", "rb") as f:
    regression_model = pickle.load(f)

def load_geojson_centroids(geojson_file):
    """Load GeoJSON centroids.

    Params:
    ------
        geojson_file (string):
            Path to geojson file that contains the centroid (representation coordinate) for country.
    """
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)
    centroids = {}
    for feature in geojson_data["features"]:
        try:
            country_name = feature["properties"]["name"]
            geo_point = feature["properties"]["geo_point_2d"]
            centroids[country_name] = (geo_point["lat"], geo_point["lon"])  # (latitude, longitude)
        except KeyError as e:
            print(f"Missing key in GeoJSON for {country_name}: {e}")
        except Exception as e:
            print(f"Error processing {country_name}: {e}")
    return centroids

# Load GeoJSON boundaries
def load_geojson_boundaries(geojson_file):
    """Load GeoJSON country shape.

    Params:
    ------
        geojson_file (string):
            Path to geojson file that contains the shape in coordinate representation for all countries in dataset.
    """
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)

    country_data = {}
    for feature in geojson_data["features"]:
        try:
            country_name = feature["properties"]["name"]
            boundary_geometry = shape(feature["geometry"])

            # Extract latitude and longitude bounds
            min_lon, min_lat, max_lon, max_lat = boundary_geometry.bounds
            country_data[country_name] = {
                "boundary": boundary_geometry,
                "lat_range": (min_lat, max_lat),
                "lon_range": (min_lon, max_lon),
            }
        except Exception as e:
            print(f"Error loading boundary for {country_name}: {e}")

    return country_data

# Load GeoJSON data
geojson_file = "data/world-administrative-boundaries.geojson"
centroids = load_geojson_centroids(geojson_file)
country_data = load_geojson_boundaries(geojson_file)

######################################## Predict ###############################################

def adjust_to_boundary(lat, lon, predicted_country, input_features):
    """This fuction uses the country's geoshape and centroid pulled from the geojson file to generate 
    a more accuracte prediction from the regression model.This function uses the predicted country's geoshape 
    to adjusted the prediction bounds of the regression model.

    Params:
    -------
        lat (float):
            Latitude coordinate value of predicted country.
        lon (float):
            Longitude coordinate value of predicted country.
        predicted_country (string):
            The predicted country from the classification model (streetCLIP).
        input_features (np.array):
            The 4 extracted features that were taken in feature_extractor.py.
            
    Returns:
    --------
        lat (float): The adjusted latitude coordinate, based off the country bounds and centroid of the country.
        lon (float): The adjusted longitude coordinate, based off the country bounds and centroid of the country.
    """
    if predicted_country in country_data:
        boundary = country_data[predicted_country]["boundary"]
        lat_range = country_data[predicted_country]["lat_range"]
        lon_range = country_data[predicted_country]["lon_range"]
        point = Point(lon, lat)

        if boundary.contains(point):
            print("Point is within the boundary.")
            return lat, lon
        else:
            print("Point is outside the boundary. Constraining regression model to country bounds...")

            def constrained_predict():
                """This function limits the predicition range of the regression model from the whole globe, 
                down to the predicted country's outer boundaries.
                
                Returns:
                --------
                    constrained_lat (float): New latitude coordinate value that is within country geoshape boundary.
                    constrained_lon (float): New longitude coordinate value that is within country geoshape boundary.
                    midpoint_lat (float): Midpoint latitude coordinate of country's geoshape boundary.
                    midpoint_lon (float): Midpoint longitude coordinate of country's geoshape boundary.
                """
                # Default to (0.0, 0.0)
                centroid = centroids.get(predicted_country, (0.0, 0.0))
                centroid_features = np.array(centroid).reshape(1, -1)
                combined_features = np.hstack([input_features, centroid_features])
                constrained_lat, constrained_lon = regression_model.predict(combined_features)[0]
                return constrained_lat, constrained_lon

            # Attempt a new prediction
            constrained_lat, constrained_lon = constrained_predict()
            constrained_point = Point(constrained_lon, constrained_lat)

            # Validate the new prediction
            if boundary.contains(constrained_point):
                print("Constrained point is within the boundary.")
                return constrained_lat, constrained_lon
            else:
                print("Constrained point still outside boundary. Forcing a midpoint adjustment.")
                midpoint_lat = (lat_range[0] + lat_range[1]) / 2
                midpoint_lon = (lon_range[0] + lon_range[1]) / 2
                print(f"Returning midpoint of country bounds: Latitude {midpoint_lat}, Longitude {midpoint_lon}")
                return midpoint_lat, midpoint_lon
    else:
        print(f"Boundary data not found for {predicted_country}.")
        return lat, lon


def predict_coordinates(image_path, predicted_country):
    """This fuction is the raw prediction of the regression model, it is based mostly off the input image.
    It also accounts somewhat for the centroid and limited features (4) from the feature extractor.

    Params:
    -------
        image_path (string):
            Relative or absoulte path to data file.
        predicted_country (string):
            The country that the regression model predicted with the most confidence.
            
    Returns:
    -------
        predicted_lat (flaot): The predicted latitude coordinate of the image.
        predicted_lon (float): The predicted longitude coordinate of the image.
        reduced_features: Makes sure the feature set is reduced down to 4. This is here to avoid frustrating errors encountered.

    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt", padding=True).to(device)

    # Extract features using the StreetCLIP model
    with torch.no_grad():
        clip_features = model.get_image_features(**inputs).cpu().numpy()

    reduced_features = clip_features[:, :4]  # Use the first 4 dimensions
    # Default to (0.0, 0.0)
    centroid = centroids.get(predicted_country, (0.0, 0.0))
    centroid_features = np.array(centroid).reshape(1, -1)

    # Combine reduced features and centroid
    combined_features = np.hstack([reduced_features, centroid_features])

    # Predict coordinates using the regression model
    predicted_lat, predicted_lon = regression_model.predict(combined_features)[0]
    return predicted_lat, predicted_lon, reduced_features
