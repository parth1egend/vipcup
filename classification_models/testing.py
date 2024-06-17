import os
import torch
from torchvision import transforms
from PIL import Image
import tqdm

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load the model from .pth file
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Function to generate labels for the test data
def generate_labels(model, test_data_folder):
    labels = {}
    for folder_name in os.listdir(test_data_folder):
        folder_path = os.path.join(test_data_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_labels = []
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)
                image = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(image)
                    _, predicted = torch.max(output, 1)
                    folder_labels.append(predicted.item())
            labels[folder_name] = folder_labels
    return labels

# Function to generate labels for the test data
def generate_labels_3stacked(model, test_data_folder):
    labels = {}
    for folder_name in tqdm(os.listdir(test_data_folder), desc="Folders"):
        folder_path = os.path.join(test_data_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_labels = []
            for image_name in tqdm(os.listdir(folder_path), desc="Images", leave=False):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path)
                image = transform(image)
                # Stack the image along the channel dimension thrice
                stacked_image = torch.stack([image, image, image], dim=0)
                with torch.no_grad():
                    output = model(stacked_image.unsqueeze(0))
                    _, predicted = torch.max(output, 1)
                    folder_labels.append(predicted.item())
            labels[folder_name] = folder_labels
    return labels


# Load your models
model_paths = ['/raid/student/2021/ai21btech11005/vipcup/classification_models/models/alexnet_3stacked_model.pth']  # paths to your model .pth files
models = [load_model(path) for path in model_paths]

# Define your test data folder
test_data_folder = '/raid/student/2021/ai21btech11005/vipcup/ICIP_test_data_final'

# Generate labels using each model
all_labels = []
for model in models:
    labels = generate_labels_3stacked(model, test_data_folder)
    all_labels.append(labels)

# Save the labels to a file
output_file = '/raid/student/2021/ai21btech11005/vipcup/classification_models/alexnet_3stacked_labels.txt'
with open(output_file, 'w') as file:
    for labels in all_labels:
        for folder_name, folder_labels in labels.items():
            file.write(f"{folder_name}: {folder_labels}\n")
