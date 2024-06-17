import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import timm
from efficientnet_pytorch import EfficientNet

# Custom Dataset class
class EyeDataset(Dataset):
    def __init__(self, root_dir, transform=None, three_channel=False):
        self.root_dir = root_dir
        self.transform = transform
        self.three_channel = three_channel
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
                            if self.three_channel:
                                image = torch.cat([image, image, image], dim=0)  # Stack to 3 channels
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

# Load entire dataset to GPU
def load_dataset_to_gpu(dataset, device):
    images = []
    labels = []
    for img, label in tqdm(dataset, desc="Loading dataset to GPU"):
        images.append(img.to(device))
        labels.append(torch.tensor(label, device=device))
    return torch.stack(images), torch.tensor(labels, device=device)

# Function to train the model
def train_model(train_images, train_labels, val_images, val_labels, model, optimizer, criterion, num_epochs, batch_size, device):
    train_performance = []
    val_performance = []
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
        train_performance.append(epoch_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(val_images), batch_size):
                images = val_images[i:i + batch_size]
                labels = val_labels[i:i + batch_size]
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_images)
        accuracy = 100. * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        val_performance.append((val_loss, accuracy))
    
    return train_performance, val_performance

# Function to modify EfficientNet for single-channel input
def modify_efficientnet_for_single_channel(model):
    model._conv_stem = nn.Conv2d(1, model._conv_stem.out_channels, kernel_size=model._conv_stem.kernel_size,
                                 stride=model._conv_stem.stride, padding=model._conv_stem.padding, bias=False)
    return model

# Function to modify MaxViT to accept single-channel input
def modify_maxvit_for_single_channel(model):
    model.stem.conv1 = nn.Conv2d(1, model.stem.conv1.out_channels, kernel_size=model.stem.conv1.kernel_size,
                                stride=model.stem.conv1.stride, padding=model.stem.conv1.padding, bias=False)
    return model

# Function to run hyperparameter tuning
def hyperparameter_tuning(model_name, modify_model_fn=None, three_channel=False):
    # Define paths
    dataset_path = "/raid/student/2021/ai21btech11005/vipcup/ICIP training data"
    
    # Define hyperparameters
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    batch_sizes = [16, 32, 64]
    optimizers = ['Adam', 'SGD']
    num_epochs = 10

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    full_dataset = EyeDataset(root_dir=dataset_path, transform=transform, three_channel=three_channel)
    train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.labels)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # Move datasets to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels = load_dataset_to_gpu(train_dataset, device)
    val_images, val_labels = load_dataset_to_gpu(val_dataset, device)

    best_loss = float('inf')
    best_accuracy = 0.0
    best_params = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for optimizer_name in optimizers:
                print(f"Training {model_name} with Learning Rate={lr}, Batch Size={batch_size}, Optimizer={optimizer_name}")

                # Load model
                if model_name == 'efficientnet_single_channel':
                    model = EfficientNet.from_name('efficientnet-b0')
                    model = modify_model_fn(model)
                elif model_name == 'efficientnet_3stacked':
                    model = EfficientNet.from_name('efficientnet-b0')
                elif model_name == 'maxvit_single_channel':
                    model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=False)
                    model = modify_model_fn(model)
                elif model_name == 'maxvit_3stacked':
                    model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=False)

                model = model.to(device)

                # Criterion and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

                # Train the model
                train_perf, val_perf = train_model(train_images, train_labels, val_images, val_labels, model, optimizer, criterion, num_epochs, batch_size, device)

                # Final validation loss and accuracy
                final_val_loss, final_val_accuracy = val_perf[-1]

                # Store the current loss with the current params in a file
                with open(f"/raid/student/2021/ai21btech11005/vipcup/classification_models/{model_name}_params_loss.txt", "a") as f:
                    f.write(f"Learning Rate: {lr}, Batch Size: {batch_size}, Optimizer: {optimizer_name}\n")
                    f.write(f"Final Validation Loss: {final_val_loss}\n")
                    f.write(f"Final Validation Accuracy: {final_val_accuracy}\n\n")

                # Check if current configuration is the best
                if final_val_loss < best_loss:
                    best_loss = final_val_loss
                    best_accuracy = final_val_accuracy
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'optimizer': optimizer_name,
                    }

    # Write the best hyperparameters and loss to a file
    with open(f"/raid/student/2021/ai21btech11005/vipcup/classification_models/{model_name}_best.txt", "w") as f:
        f.write(f"Best hyperparameters:\n{best_params}\n")
        f.write(f"Best Validation Loss: {best_loss}\n")
        f.write(f"Best Accuracy: {best_accuracy}\n")

# Main function
def main():
    # hyperparameter_tuning('efficientnet_single_channel', modify_efficientnet_for_single_channel, three_channel=False)
    # hyperparameter_tuning('efficientnet_3stacked', three_channel=True)
    hyperparameter_tuning('maxvit_single_channel', modify_maxvit_for_single_channel, three_channel=False)
    hyperparameter_tuning('maxvit_3stacked', three_channel=True)

if __name__ == "__main__":
    main()
