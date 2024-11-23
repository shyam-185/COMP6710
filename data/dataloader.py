import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Define paths
image_directory = "data/streetviews"

# Extract image paths and parse latitude/longitude
data = []
for root, dirs, files in os.walk(image_directory):
    for file in files:
        if file.endswith(".jpg"):
            try:
                lat, lon = map(float, file.replace(".jpg", "").split(","))
                file_path = os.path.join(root, file)
                data.append({'image': file_path, 'latitude': lat, 'longitude': lon})
            except ValueError:
                print(f"Skipping invalid file: {file}")

# Create a DataFrame
metadata = pd.DataFrame(data)
metadata.to_csv("streetview_metadata.csv", index=False)
print("Metadata created and saved.")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Dataset
class StreetViewDataset(Dataset):
    def __init__(self, metadata, transform):
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        path = self.metadata.iloc[idx]['image']
        lat, lon = self.metadata.iloc[idx][['latitude', 'longitude']]
        try:
            img = Image.open(path).convert('RGB')
        except (IOError, FileNotFoundError) as e:
            print(f"Error loading image {path}: {e}")
            return None, None
        img = self.transform(img)
        return img, torch.tensor([lat, lon], dtype=torch.float32)

# Create DataLoader
dataset = StreetViewDataset(metadata, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Test if DataLoader works
# for i, (images, coords) in enumerate(dataloader):
#     print(f"Batch {i}:")
#     print(f"Images: {images.size()}, Coordinates: {coords.size()}")
#     break

# Visualize data distribution
# plt.hist2d(metadata['longitude'], metadata['latitude'], bins=50, cmap='viridis')
# plt.colorbar(label='Frequency')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Dataset Distribution')
# plt.show()
