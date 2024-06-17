import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
import pandas as pd
import timm
from efficientnet_pytorch import EfficientNet

# Define paths
final_test_data_path = '/raid/student/2021/ai21btech11005/vipcup/BasicSR/results/EDSR_TEST_INFER'
model_folder_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models'

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom Dataset class for final test data
class FinalTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff','tif')):
                    self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except (UnidentifiedImageError, IOError) as e:
            print(f"Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

# Load final test dataset
final_test_dataset = FinalTestDataset(final_test_data_path, transform=transform)
final_test_loader = DataLoader(final_test_dataset, batch_size=32, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to modify models to accept single-channel input
def modify_resnet_for_single_channel(model):
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    return model

def modify_vit_for_single_channel(model):
    model.patch_embed.proj = nn.Conv2d(1, model.patch_embed.proj.out_channels, kernel_size=model.patch_embed.proj.kernel_size,
                                       stride=model.patch_embed.proj.stride, padding=model.patch_embed.proj.padding, bias=False)
    return model

def modify_maxvit_for_single_channel(model):
    model.stem.conv1 = nn.Conv2d(1, model.stem.conv1.out_channels, kernel_size=model.stem.conv1.kernel_size,
                                 stride=model.stem.conv1.stride, padding=model.stem.conv1.padding, bias=False)
    return model

def modify_efficientnet_for_single_channel(model):
    model._conv_stem = nn.Conv2d(1, model._conv_stem.out_channels, kernel_size=model._conv_stem.kernel_size,
                                 stride=model._conv_stem.stride, padding=model._conv_stem.padding, bias=False)
    return model

def modify_alexnet_for_single_channel(model):
    model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                  stride=model.features[0].stride, padding=model.features[0].padding)
    return model

# Define model loading function
def load_model(model_name, weights_path):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model = modify_resnet_for_single_channel(model)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        model = modify_resnet_for_single_channel(model)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model = modify_resnet_for_single_channel(model)
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model = modify_vit_for_single_channel(model)
    elif model_name == 'maxvit':
        model = timm.create_model('maxvit_tiny_tf_224', pretrained=False)
        model = modify_maxvit_for_single_channel(model)
    elif model_name == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-b0')
        model = modify_efficientnet_for_single_channel(model)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=None)
        model = modify_alexnet_for_single_channel(model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Define function to predict and store labels patient-wise
def predict_and_store_labels_patient_wise(model, loader, model_name):
    predictions = {}
    
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for path, pred in zip(paths, preds.cpu().numpy()):
                patient_folder = os.path.basename(os.path.dirname(path))
                if patient_folder not in predictions:
                    predictions[patient_folder] = []
                predictions[patient_folder].append((os.path.basename(path), pred))
    
    # Store predictions patient-wise
    for patient, preds in predictions.items():
        df = pd.DataFrame(preds, columns=['image_name', 'predicted_label'])
        df.to_csv(f'predictions_EDSR/final_predictions_{model_name}_{patient}.csv', index=False)

# Define paths to weights
model_weights = {
    'resnet18': os.path.join(model_folder_path, 'resnet18_single_channel_model.pth'),
    'resnet34': os.path.join(model_folder_path, 'resnet34_single_channel_model.pth'),
    'resnet50': os.path.join(model_folder_path, 'resnet50_single_channel_model.pth'),
    # 'vit': os.path.join(model_folder_path, 'vit_single_channel_model.pth'),
    # 'maxvit': os.path.join(model_folder_path, 'maxvit_single_channel_model.pth'),
    # 'efficientnet': os.path.join(model_folder_path, 'efficient_single_channel_model.pth'),
    'alexnet': os.path.join(model_folder_path, 'alexnet_single_channel_model.pth'),
}

# Evaluate all models on final test dataset
for model_name, weights_path in model_weights.items():
    print(f"Running predictions for {model_name}...")
    model = load_model(model_name, weights_path)
    predict_and_store_labels_patient_wise(model, final_test_loader, model_name)
    print(f"Predictions for {model_name} completed.")
