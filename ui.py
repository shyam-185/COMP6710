import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium
import numpy as np
import json
from shapely.geometry import shape, Point
from regression_model import classify, predict_coordinates, adjust_to_boundary, load_geojson_boundaries

##############################################################################################
#                                                                                            #
#                                      UI Demo                                               #
#                                                                                            #
##############################################################################################

# I used chatgpt on some aspect of this code to help with figuring out errors, especially when displaying
# Initialize
country_data = load_geojson_boundaries("data/world-administrative-boundaries.geojson")
st.title("Geolocation Prediction Demo Using Clasification and Regression")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Uploaded Image Classification
    image = Image.open(uploaded_file).convert("RGB")
    country_scores = classify(image)
    predicted_country, confidence = max(country_scores.items(), key=lambda item: item[1])
    confidence *= 100

    st.write(f"The country where this image was taken is most probably:")
    st.write(f"**{predicted_country}: {confidence:.2f}% Confidence**")

    # Uploaded Image Regression
    predicted_coordinates = predict_coordinates(uploaded_file, predicted_country)
    predicted_lat, predicted_lon, input_features = predicted_coordinates

    adjusted_coordinates = adjust_to_boundary(predicted_lat, predicted_lon, predicted_country, input_features)
    adjusted_lat, adjusted_lon = adjusted_coordinates

    # Display Results
    st.write("### Model Predictions")
    st.write(f"- **Predicted Coordinates**: Latitude: `{predicted_lat}`, Longitude: `{predicted_lon}`")

    if (predicted_lat, predicted_lon) != (adjusted_lat, adjusted_lon):
        st.write(f"- **Adjusted Coordinates**: Latitude: `{adjusted_lat}`, Longitude: `{adjusted_lon}`")
    else:
        st.write("- **No adjustment needed**: The point is within the boundary.")

    st.write("### Visualization on Map")
    map_overlay = folium.Map(location=[adjusted_lat, adjusted_lon], zoom_start=4)

    # Predicted Marker
    folium.Marker(
        location=[predicted_lat, predicted_lon],
        popup="Predicted Coordinates",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(map_overlay)

    # Adjusted Marker
    folium.Marker(
        location=[adjusted_lat, adjusted_lon],
        popup="Adjusted Coordinates",
        icon=folium.Icon(color="green", icon="ok-sign")
    ).add_to(map_overlay)

    st_data = st_folium(map_overlay, width=725)

    # Country boundary
    if predicted_country in country_data:
        boundary = country_data[predicted_country]["boundary"]
        geojson_boundary = json.dumps({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": boundary.__geo_interface__,
                "properties": {"name": predicted_country}
            }]
        })
        folium.GeoJson(geojson_boundary, name="Country Boundary").add_to(map_overlay)

    st_folium(map_overlay, width=700, height=500)
