# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms, models
# from PIL import Image
# from tqdm import tqdm
# from collections import defaultdict
# from datetime import datetime
# from sklearn.model_selection import train_test_split

# # Custom Dataset class
# class EyeDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
#         self.image_paths = []
#         self.labels = []
#         self.images = []
#         print("Loading dataset...")
#         for label in self.classes:
#             class_dir = os.path.join(root_dir, label)
#             for subdir in os.listdir(class_dir):
#                 subdir_path = os.path.join(class_dir, subdir)
#                 if os.path.isdir(subdir_path):
#                     for img_name in os.listdir(subdir_path):
#                         img_path = os.path.join(subdir_path, img_name)
#                         image = Image.open(img_path).convert("L")
#                         if self.transform:
#                             image = self.transform(image)
#                             image = torch.cat([image, image, image], dim=0)  # Stack to 3 channels
#                         self.images.append(image)
#                         self.image_paths.append(img_path)
#                         self.labels.append(int(label))
#         print(f"Dataset loaded. Total images: {len(self.image_paths)}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         return image, label

# # Load entire dataset to GPU
# def load_dataset_to_gpu(dataset, device):
#     images = []
#     labels = []
#     for img, label in tqdm(dataset, desc="Loading dataset to GPU"):
#         images.append(img.to(device))
#         labels.append(torch.tensor(label, device=device))
#     return torch.stack(images), torch.tensor(labels, device=device)

# # Global variables for the best parameters
# best_loss = float('inf')
# best_accuracy = 0.0
# best_params = {}

# # Function to train the model with given hyperparameters
# def train_model(dataset_path, learning_rate, batch_size, optimizer_name):
#     global best_loss, best_accuracy, best_params

#     # Data transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # Load dataset
#     full_dataset = EyeDataset(root_dir=dataset_path, transform=transform)

#     # Split dataset into training and validation sets
#     train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.labels)
#     train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
#     val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

#     # Move datasets to GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_images, train_labels = load_dataset_to_gpu(train_dataset, device)
#     val_images, val_labels = load_dataset_to_gpu(val_dataset, device)

#     # Load AlexNet model
#     model = models.alexnet(pretrained=False)
#     model = model.to(device)

#     if torch.cuda.is_available():
#         print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
#         print(f"Current GPU: {torch.cuda.current_device()}")
#         print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
#     else:
#         print("CUDA is not available. Using CPU.")

#     # Criterion and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

#     # Training loop
#     num_epochs = 10  # Epochs should be 10
#     val_loss = 0.0
#     accuracy = 0.0
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         print(f"Starting epoch {epoch+1}/{num_epochs}...")
#         for i in tqdm(range(0, len(train_images), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
#             images = train_images[i:i + batch_size]
#             labels = train_labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)

#         epoch_loss = running_loss / len(train_images)
#         print(f"Epoch [{epoch+1}/{num_epochs}] complete. Training Loss: {epoch_loss:.4f}")

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for i in range(0, len(val_images), batch_size):
#                 images = val_images[i:i + batch_size]
#                 labels = val_labels[i:i + batch_size]
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item() * images.size(0)
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()

#         val_loss /= len(val_images)
#         accuracy = 100. * correct / total
#         print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

#     current_params = {
#         'learning_rate': learning_rate,
#         'batch_size': batch_size,
#         'optimizer': optimizer_name,
#     }

#     # Store the current loss with the current params in a file
#     with open("/raid/student/2021/ai21btech11005/vipcup/classification_models/alexnet_3stacked_params_loss.txt", "a") as f:
#         f.write(f"Current Validation Loss: {val_loss}\n")
#         f.write(f"Current hyperparameters: {current_params}\n\n")
#         f.write(f"Current Accuracy: {accuracy}\n")

#     # Check if current configuration is the best
#     if val_loss < best_loss:
#         best_loss = val_loss
#         best_accuracy = accuracy
#         best_params = current_params

#     # Write the best hyperparameters and loss to a file
#     with open("/raid/student/2021/ai21btech11005/vipcup/classification_models/alexnet_3stacked_params_loss.txt", "a") as f:
#         f.write(f"Best hyperparameters till now: {best_params}\n")
#         f.write(f"Best Validation Loss till now: {best_loss}\n")
#         f.write(f"Best Accuracy till now: {best_accuracy}\n")

#     return best_loss, best_accuracy

# # Main function
# def main():
#     dataset_path = "/raid/student/2021/ai21btech11005/vipcup/ICIP training data"
#     hyperparameters = {
#         'learning_rate': [1e-4],
#         'batch_size': [64],
#         'optimizer': ['Adam']
#     }
#     results = defaultdict(list)

#     # # Iterate over hyperparameters
#     # for lr in hyperparameters['learning_rate']:
#     #     for batch_size in hyperparameters['batch_size']:
#     #         for optimizer_name in hyperparameters['optimizer']:
#     #             print(f"Training with hyperparameters: Learning Rate={lr}, Batch Size={batch_size}, Optimizer={optimizer_name}")
                
#     #             # Training
#     #             loss, accuracy = train_model(dataset_path, lr, batch_size, optimizer_name)

#     #             # Record results
#     #             results['Learning Rate'].append(lr)
#     #             results['Batch Size'].append(batch_size)
#     #             results['Optimizer'].append(optimizer_name)
#     #             results['Loss'].append(loss)
#     #             results['Accuracy'].append(accuracy)
    
#     lr = hyperparameters['learning_rate'][0]
#     batch_size = hyperparameters['batch_size'][0]
#     optimizer_name = hyperparameters['optimizer'][0]

#     print(f"Training with hyperparameters: Learning Rate={lr}, Batch Size={batch_size}, Optimizer={optimizer_name}")
    
#     # Training
#     loss, accuracy = train_model(dataset_path, lr, batch_size, optimizer_name)

#     # Record results
#     results['Learning Rate'].append(lr)
#     results['Batch Size'].append(batch_size)
#     results['Optimizer'].append(optimizer_name)
#     results['Loss'].append(loss)
#     results['Accuracy'].append(accuracy)

#     # Analyze results
#     best_index = results['Loss'].index(min(results['Loss']))
#     best_hyperparameters = {
#         'Learning Rate': results['Learning Rate'][best_index],
#         'Batch Size': results['Batch Size'][best_index],
#         'Optimizer': results['Optimizer'][best_index],
#         'Loss': results['Loss'][best_index],
#         'Accuracy': results['Accuracy'][best_index]
#     }
#     # Store best hyperparameters in result file
#     with open("/raid/student/2021/ai21btech11005/vipcup/classification_models/alexnet_3stacked_best.txt", "w") as f:
#         f.write("Best Hyperparameters:\n")
#         f.write(f"Learning Rate: {best_hyperparameters['Learning Rate']}\n")
#         f.write(f"Batch Size: {best_hyperparameters['Batch Size']}\n")
#         f.write(f"Optimizer: {best_hyperparameters['Optimizer']}\n")
#         f.write(f"Loss: {best_hyperparameters['Loss']}\n")
#         f.write(f"Accuracy: {best_hyperparameters['Accuracy']}\n")

# if __name__ == "__main__":
#     main()


# -------------------------
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, Subset
# from torchvision import transforms, models
# from PIL import Image
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import numpy as np
# import random

# # Custom Dataset class
# class EyeDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
#         self.image_paths = []
#         self.labels = []
#         self.images = []
#         print("Loading dataset...")
#         for label in self.classes:
#             class_dir = os.path.join(root_dir, label)
#             for subdir in os.listdir(class_dir):
#                 subdir_path = os.path.join(class_dir, subdir)
#                 if os.path.isdir(subdir_path):
#                     for img_name in os.listdir(subdir_path):
#                         img_path = os.path.join(subdir_path, img_name)
#                         image = Image.open(img_path).convert("L")
#                         if self.transform:
#                             image = self.transform(image)
#                             image = torch.cat([image, image, image], dim=0)  # Stack to 3 channels
#                         self.images.append(image)
#                         self.image_paths.append(img_path)
#                         self.labels.append(int(label))
#         print(f"Dataset loaded. Total images: {len(self.image_paths)}")

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         return image, label

# # Load entire dataset to GPU
# def load_dataset_to_gpu(dataset, device):
#     images = []
#     labels = []
#     for img, label in tqdm(dataset, desc="Loading dataset to GPU"):
#         images.append(img.to(device))
#         labels.append(torch.tensor(label, device=device))
#     return torch.stack(images), torch.tensor(labels, device=device)

# # Function to train the model
# def train_model(train_images, train_labels, model, optimizer, criterion, num_epochs, batch_size, device):
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         print(f"Starting epoch {epoch+1}/{num_epochs}...")
#         for i in tqdm(range(0, len(train_images), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
#             images = train_images[i:i + batch_size]
#             labels = train_labels[i:i + batch_size]
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)

#         epoch_loss = running_loss / len(train_images)
#         print(f"Epoch [{epoch+1}/{num_epochs}] complete. Training Loss: {epoch_loss:.4f}")

#     return model

# # Function to test the model
# def test_model(test_images, test_labels, model, criterion, batch_size, device):
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for i in range(0, len(test_images), batch_size):
#             images = test_images[i:i + batch_size]
#             labels = test_labels[i:i + batch_size]
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item() * images.size(0)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     test_loss /= len(test_images)
#     accuracy = 100. * correct / total
#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
#     return test_loss, accuracy

# # Set random seed for reproducibility
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Main function
# def main():
#     set_seed(42)
#     # Define paths
#     train_dataset_path = "/raid/student/2021/ai21btech11005/vipcup/ICIP training data"

#     # Define hyperparameters
#     learning_rate = 1e-4
#     batch_size = 64
#     num_epochs = 100
#     optimizer_name = 'Adam'

#     # Data transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # Load training dataset
#     full_dataset = EyeDataset(root_dir=train_dataset_path, transform=transform)

#     # Split dataset based on subfolders
#     subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(train_dataset_path) for d in dirs]
#     train_subdirs, test_subdirs = train_test_split(subdirs, test_size=0.2, random_state=42)
    
#     train_idx = [i for i, path in enumerate(full_dataset.image_paths) if any(sd in path for sd in train_subdirs)]
#     test_idx = [i for i, path in enumerate(full_dataset.image_paths) if any(sd in path for sd in test_subdirs)]
    
#     # # Equal spliting of data using train_test_split
    
#     # train_idx, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, stratify=full_dataset.labels)

    
    
    
    

#     train_dataset = Subset(full_dataset, train_idx)
#     test_dataset = Subset(full_dataset, test_idx)

#     # Print subfolder names for verification
#     print("\nTraining subfolders:")
#     for sd in train_subdirs[:5]:
#         print(sd)
    
#     print("\nTesting subfolders:")
#     for sd in test_subdirs[:5]:
#         print(sd)

#     # Load datasets to GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_images, train_labels = load_dataset_to_gpu(train_dataset, device)
#     test_images, test_labels = load_dataset_to_gpu(test_dataset, device)

#     # Load AlexNet model
#     model = models.alexnet(pretrained=False)
#     model = model.to(device)

#     # Criterion and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

#     # Train the model on the initial dataset
#     print("Training on the initial dataset...")
#     model = train_model(train_images, train_labels, model, optimizer, criterion, num_epochs, batch_size, device)

#     # Save the trained model
#     torch.save(model.state_dict(), "/raid/student/2021/ai21btech11005/vipcup/classification_models/models/alexnet_3stacked_model.pth")
#     print("Model saved.")

#     # Test the model on the test dataset
#     print("Testing on the test dataset...")
#     test_loss, test_accuracy = test_model(test_images, test_labels, model, criterion, batch_size, device)

#     # Save performance metrics to a text file
#     with open("/raid/student/2021/ai21btech11005/vipcup/classification_models/testing/alexnet_3stacked_testing.txt", "w") as f:
#         f.write(f"Test Loss: {test_loss}\n")
#         f.write(f"Test Accuracy: {test_accuracy:.4f}%\n")

# if __name__ == "__main__":
#     main()




# -----------
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
    learning_rate = 1e-4
    batch_size = 64
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

    # Load AlexNet model
    model = models.alexnet(pretrained=False)
    model = model.to(device)

    # Criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # Start MLflow run
    # mlflow.set_tracking_uri("/raid/student/2021/ai21btech11005/vipcup/classification_models/mlruns")  # Set tracking URI
    # mlflow.set_experiment("Alexnet 3 Channel")  # Set experiment name
    # mlflow.autolog()
    # with mlflow.start_run(run_name="AlexNet 3 Channel Training and Testing") as run:
        # Log parameters
        # mlflow.log_param("learning_rate", learning_rate)
        # mlflow.log_param("batch_size", batch_size)
        # mlflow.log_param("num_epochs", num_epochs)
        # mlflow.log_param("optimizer", optimizer_name)

    # Train the model
    print("Training on the initial dataset...")
    model = train_model(train_images, train_labels, model, optimizer, criterion, num_epochs, batch_size, device)

    # Save the trained model
    torch.save(model.state_dict(), "/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/alexnet_3stacked_model.pth")
    # mlflow.pytorch.log_model(model, "model")
    print("Model saved.")

    # Test the model
    print("Testing on the test dataset...")
    test_loss, test_accuracy = test_model(test_images, test_labels, model, criterion, batch_size, device)

    # Log performance metrics
    # mlflow.log_metric("Test Loss", test_loss)
    # mlflow.log_metric("Test Accuracy", test_accuracy)

    # Save performance metrics to a text file
    with open("/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/testing/alexnet_3stacked_testing.txt", "w") as f:
        f.write(f"Test Loss: {test_loss}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}%\n")

if __name__ == "__main__":
    main()
