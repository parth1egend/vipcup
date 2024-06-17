import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import timm
from efficientnet_pytorch import EfficientNet



# Define paths
train_dataset_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/EDSR_Data'

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
full_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)

# Split dataset based on subfolders
subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(train_dataset_path) for d in dirs]
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

# Define evaluation function
def evaluate_model(model, loader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Compute specificity
    cm = confusion_matrix(all_labels, all_preds)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    specificity = tn / (tn + fp)
    
    return accuracy, precision, recall, specificity, f1

# Define paths to weights
model_weights = {
    'resnet50': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/resnet50_single_channel_model.pth',
    'resnet34': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/resnet34_single_channel_model.pth',
    'resnet18': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/resnet18_single_channel_model.pth',
    # 'vit': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/vit_single_channel_model.pth',
    # 'maxvit': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/maxvit_single_channel_model.pth',
    'alexnet': '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/alexnet_single_channel_model.pth',
}

# Evaluate all models
with open('/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/metrics_EDSR.txt', 'a') as f:
    for model_name, weights_path in model_weights.items():
        print("Loading model:", model_name)
        model = load_model(model_name, weights_path)
        print("Evaluating model:", model_name)
        accuracy, precision, recall, specificity, f1 = evaluate_model(model, test_loader)
        
        print(f"Metrics for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print()
        f.write(f"Metrics for {model_name}:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\n")
