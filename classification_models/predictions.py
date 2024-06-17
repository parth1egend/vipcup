import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import timm
from efficientnet_pytorch import EfficientNet

# Set cudnn benchmark and deterministic to handle potential CUDNN exceptions
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Define paths
train_dataset_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/SR_Data'
model_folder_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models_random_split'

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom ImageFolder class to handle corrupted images
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super(CustomImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, IOError) as e:
            print(f"Skipping corrupted image: {self.imgs[index][0]}")
            return self.__getitem__(index + 1)

# Load dataset
full_dataset = CustomImageFolder(train_dataset_path, transform=transform)

# Split dataset based on subfolders
subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(train_dataset_path) for d in dirs if d in ['0', '1', '2']]
train_subdirs, test_subdirs = train_test_split(subdirs, test_size=0.2, random_state=42)

train_idx = [i for i, path in enumerate(full_dataset.imgs) if any(sd in path[0] for sd in train_subdirs)]
test_idx = [i for i, path in enumerate(full_dataset.imgs) if any(sd in path[0] for sd in test_subdirs)]

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

# Define DataLoader
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to modify ResNet to accept single-channel input
def modify_resnet_for_single_channel(model):
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    return model

# Function to modify ViT to accept single-channel input
def modify_vit_for_single_channel(model):
    model.patch_embed.proj = nn.Conv2d(1, model.patch_embed.proj.out_channels, kernel_size=model.patch_embed.proj.kernel_size,
                                       stride=model.patch_embed.proj.stride, padding=model.patch_embed.proj.padding, bias=False)
    return model

# Function to modify MaxViT to accept single-channel input
def modify_maxvit_for_single_channel(model):
    model.stem.conv1 = nn.Conv2d(1, model.stem.conv1.out_channels, kernel_size=model.stem.conv1.kernel_size,
                                 stride=model.stem.conv1.stride, padding=model.stem.conv1.padding, bias=False)
    return model

# Function to modify EfficientNet to accept single-channel input
def modify_efficientnet_for_single_channel(model):
    model._conv_stem = nn.Conv2d(1, model._conv_stem.out_channels, kernel_size=model._conv_stem.kernel_size,
                                 stride=model._conv_stem.stride, padding=model._conv_stem.padding, bias=False)
    return model

# Function to modify AlexNet to accept single-channel input
def modify_alexnet_for_single_channel(model):
    model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                  stride=model.features[0].stride, padding=model.features[0].padding)
    return model

# Define model loading function
def load_model(model_name, weights_path):
    if model_name == 'resnet18':
        model = models.resnet18()
        model = modify_resnet_for_single_channel(model)
    elif model_name == 'resnet34':
        model = models.resnet34()
        model = modify_resnet_for_single_channel(model)
    elif model_name == 'resnet50':
        model = models.resnet50()
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

# Define evaluation function
def evaluate_model(model, loader, model_name):
    all_preds = []
    all_labels = []
    all_image_names = []
    patient_predictions = {}
    
    with torch.no_grad():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_image_names.extend(paths)
            
            for path, pred in zip(paths, preds.cpu().numpy()):
                patient_folder = os.path.basename(os.path.dirname(path))
                if patient_folder not in patient_predictions:
                    patient_predictions[patient_folder] = []
                patient_predictions[patient_folder].append((os.path.basename(path), labels.cpu().numpy()[0], pred))
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Compute specificity
    cm = confusion_matrix(all_labels, all_preds)
    tn = cm[0, 0] if cm.shape[0] > 1 else 0
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    tp = cm[1, 1] if cm.shape[1] > 1 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Store predictions with image names patient-wise
    for patient, preds in patient_predictions.items():
        df = pd.DataFrame(preds, columns=['image_name', 'true_label', 'predicted_label'])
        df.to_csv(f'predictions_{model_name}_{patient}.csv', index=False)
    
    return accuracy, precision, recall, specificity, f1

# Define paths to weights
model_weights = {
    'resnet18': os.path.join(model_folder_path, 'resnet18_single_channel_model.pth'),
    'resnet34': os.path.join(model_folder_path, 'resnet34_single_channel_model.pth'),
    'resnet50': os.path.join(model_folder_path, 'resnet50_single_channel_model.pth'),
    'vit': os.path.join(model_folder_path, 'vit_single_channel_model.pth'),
    'maxvit': os.path.join(model_folder_path, 'maxvit_single_channel_model.pth'),
    'efficientnet': os.path.join(model_folder_path, 'efficient_single_channel_model.pth'),
    'alexnet': os.path.join(model_folder_path, 'alexnet_single_channel_model.pth'),
}

# Custom DataLoader to return image paths
class CustomDataLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            images, labels = batch
            paths = [self.dataset.dataset.imgs[idx][0] for idx in self.dataset.indices]
            yield images, labels, paths

# Use the custom DataLoader
test_loader = CustomDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate all models
for model_name, weights_path in model_weights.items():
    model = load_model(model_name, weights_path)
    accuracy, precision, recall, specificity, f1 = evaluate_model(model, test_loader, model_name)
    
    with open('metrics.txt', 'a') as file:
        file.write(f"Metrics for {model_name}:\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall (Sensitivity): {recall:.4f}\n")
        file.write(f"Specificity: {specificity:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write("\n")
