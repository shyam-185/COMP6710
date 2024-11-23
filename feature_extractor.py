from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA

##############################################################################################
#                                                                                            #
#                                    Feature Extractor                                       #
#                                                                                            #
##############################################################################################

# Load StreetCLIP (Classification model)
# Trained on an original dataset of 1.1 million street-level urban and rural geo-tagged images
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

def reduce_dimensions(features, n_components=4):
"""Reduces the dimensionality of feature embeddings using PCA.

This function is used to reduce the number of features from 768 to a smaller number 4 to make the data more manageable for regression.
I did this because of the limited memory on my machine and save a lot of time.

Params:
-------
    features (np.ndarray): 
        Input feature embeddings of shape (n_samples, n_features), 
    n_components (int, optional): 
        The number of principal components to retain. Defaults to 4.

Return:
-------
    np.ndarray: Reduced feature embeddings of shape (n_samples, n_components).
"""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


def preprocess_images(image_paths):
"""Preprocessing function conforming to StreetCLIP which uses Open AI's CLIP ViT.

Params:
------
    image_paths (list): 
        List of image file paths.
        
Notes:
------
    Open AI's CLIP ViT uses 14x14 pixel patches and images with a 336 pixel side length.
    https://huggingface.co/geolocal/StreetCLIP
"""
    processed_images = []
    for image in image_paths:
        img = Image.open(image).convert("RGB")
        img = img.resize((336, 336))
        processed_images.append(img)
    return processed_images

def extract_features(image_paths, batch_size=16):
"""Extracts features for provided list of image paths in batches.

Params:
-----
    image_paths (list): 
        List of image file paths.
    batch_size (int): 
        Number of images to process in a single batch.

Return:
--------
    np.ndarray: Feature embeddings for all images.
"""
    feature_embeddings = []

    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = batch_start + batch_size
        batch_paths = image_paths[batch_start:batch_end]

        processed_images = preprocess_images(batch_paths)
        inputs = processor(images=processed_images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            batch_features = model.get_image_features(**inputs)
            feature_embeddings.append(batch_features.cpu().numpy())
    return np.concatenate(feature_embeddings, axis=0)

######################################## Main #########################################

if __name__ == "__main__":
    metadata = pd.read_csv("data/streetview_metadata.csv")
    image_paths = metadata['image'].tolist()

    # Extract and save features
    features = extract_features(image_paths, batch_size=16)
    reduced_features = reduce_dimensions(features, n_components=4)
    np.save("data/streetclip_features.npy", reduced_features)
