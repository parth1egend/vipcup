import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
import mlflow
import mlflow.pytorch

# Custom Dataset class
class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        self.labels = []
        self.images = []
        print("Loading dataset...")
        for label in self.classes:
            class_dir = os.path.join(root_dir, label)
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_name in os.listdir(subdir_path):
                        img_path = os.path.join(subdir_path, img_name)
                        image = Image.open(img_path).convert("L")
                        if self.transform:
                            image = self.transform(image)
                        self.images.append(image)
                        self.image_paths.append(img_path)
                        self.labels.append(int(label))
        print(f"Dataset loaded. Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Function to modify ResNet to accept single-channel input
def modify_resnet_for_single_channel(model):
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    return model

# Load entire dataset to GPU
def load_dataset_to_gpu(dataset, device):
    images = []
    labels = []
    for img, label in tqdm(dataset, desc="Loading dataset to GPU"):
        images.append(img.to(device))
        labels.append(torch.tensor(label, device=device))
    return torch.stack(images), torch.tensor(labels, device=device)

# Function to train the model
def train_model(train_images, train_labels, model, optimizer, criterion, num_epochs, batch_size, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        for i in tqdm(range(0, len(train_images), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = train_images[i:i + batch_size]
            labels = train_labels[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_images)
        print(f"Epoch [{epoch+1}/{num_epochs}] complete. Training Loss: {epoch_loss:.4f}")
        # mlflow.log_metric("Training Loss", epoch_loss, step=epoch)

    return model

# Function to test the model
def test_model(test_images, test_labels, model, criterion, batch_size, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            images = test_images[i:i + batch_size]
            labels = test_labels[i:i + batch_size]
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss /= len(test_images)
    accuracy = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    # mlflow.log_metric("Test Loss", test_loss)
    # mlflow.log_metric("Test Accuracy", accuracy)
    return test_loss, accuracy

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main function
def main():
    set_seed(42)
    # Define paths
    train_dataset_path = "/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/EDSR_Data"

    # Define hyperparameters
    learning_rate = 1e-3
    batch_size = 16
    num_epochs = 100
    optimizer_name = 'Adam'

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load training dataset
    full_dataset = EyeDataset(root_dir=train_dataset_path, transform=transform)

    # Split dataset based on subfolders
    subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(train_dataset_path) for d in dirs]
    train_subdirs, test_subdirs = train_test_split(subdirs, test_size=0.2, random_state=42)
    
    train_idx = [i for i, path in enumerate(full_dataset.image_paths) if any(sd in path for sd in train_subdirs)]
    test_idx = [i for i, path in enumerate(full_dataset.image_paths) if any(sd in path for sd in test_subdirs)]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Print subfolder names for verification
    print("\nTraining subfolders:")
    for sd in train_subdirs[:5]:
        print(sd)
    
    print("\nTesting subfolders:")
    for sd in test_subdirs[:5]:
        print(sd)
    
    
    # # Split dataset randomly using train_test_split
    # train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

    # Load datasets to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels = load_dataset_to_gpu(train_dataset, device)
    test_images, test_labels = load_dataset_to_gpu(test_dataset, device)

    # Load and modify ResNet model
    model = models.resnet34(pretrained=False)
    model = modify_resnet_for_single_channel(model)
    model = model.to(device)

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Start MLflow run
    # mlflow.set_tracking_uri("/raid/student/2021/ai21btech11005/vipcup/classification_models/mlruns")  # Set tracking URI
    # mlflow.set_experiment("Resnet34 Single-Channel")  # Set experiment name
    # mlflow.autolog()
    # with mlflow.start_run(run_name="Resnet34 Single-Channel Training and Testing") as run:
        # Log parameters
        # mlflow.log_param("learning_rate", learning_rate)
        # mlflow.log_param("batch_size", batch_size)
        # mlflow.log_param("num_epochs", num_epochs)
        # mlflow.log_param("optimizer", optimizer_name)

    # Train the model
    print("Training on the initial dataset...")
    model = train_model(train_images, train_labels, model, optimizer, criterion, num_epochs, batch_size, device)

    # Save the trained model
    torch.save(model.state_dict(), "/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/resnet34_single_channel_model.pth")
    # mlflow.pytorch.log_model(model, "model")
    print("Model saved.")

    # Test the model
    print("Testing on the test dataset...")
    test_loss, test_accuracy = test_model(test_images, test_labels, model, criterion, batch_size, device)

    # Log performance metrics
    # mlflow.log_metric("Test Loss", test_loss)
    # mlflow.log_metric("Test Accuracy", test_accuracy)

    # Save performance metrics to a text file
    with open("/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/testing/resnet34_single_channel_testing.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}%\n")
if __name__ == "__main__":
    main()
