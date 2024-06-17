import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models

# Custom Dataset class to handle patient folders
class PatientDataset(Dataset):
    def __init__(self, patient_folder, transform=None):
        self.patient_folder = patient_folder
        self.transform = transform
        self.image_paths = [os.path.join(patient_folder, img) for img in os.listdir(patient_folder) if img.endswith(('png', 'jpg', 'jpeg', 'bmp','tiff'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Function to convert EfficientNet to handle single-channel input
def convert_to_single_channel(model):
    # Modify the first convolutional layer to accept single-channel input
    model._conv_stem = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
    return model

# Function to modify ResNet to accept single-channel input
def modify_resnet_for_single_channel(model):
    model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    return model

# Function to modify AlexNet for single-channel input
def modify_alexnet_for_single_channel(model):
    model.features[0] = nn.Conv2d(1, model.features[0].out_channels, kernel_size=model.features[0].kernel_size,
                                  stride=model.features[0].stride, padding=model.features[0].padding)
    return model

# Function to evaluate patient-level accuracy
def evaluate_patient_level_accuracy(model, dataset_path):
    model.eval()
    total_patients = 0
    correct_patients = 0

    parent_folders = ['0', '1', '2']  # Assuming parent folders are '0', '1', '2'

    patient_predictions = {}

    with torch.no_grad():
        for parent_folder in parent_folders:
            sub_folder = os.path.join(dataset_path, parent_folder)

            patient_folders = [os.path.join(sub_folder, patient_folder) for patient_folder in os.listdir(sub_folder) if os.path.isdir(os.path.join(sub_folder, patient_folder))]

            parent_label = int(parent_folder)  # Convert parent folder name to integer

            # for patient_folder in tqdm(patient_folders, desc=f'Evaluating Patient-level Accuracy for label {parent_label}'):
            for patient_folder in patient_folders:
                patient_dataset = PatientDataset(patient_folder, transform=transforms.Compose([
                    transforms.Resize((224, 224)),  # Resize as necessary
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485], std=[0.229]),  # Example normalization for single-channel images
                ]))
                patient_loader = DataLoader(patient_dataset, batch_size=32, shuffle=False)

                patient_preds = []
                for images, _ in patient_loader:
                    images = images.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    patient_preds.extend(predicted.cpu().numpy())
                    
                print(f"Size of patient_preds: {len(patient_preds)}")

                if patient_preds:
                    # Calculate majority prediction
                    majority_label = max(set(patient_preds), key=patient_preds.count)
                    print(f'Patient folder: {os.path.basename(patient_folder)}, True label: {parent_label}, Predicted label: {majority_label}')

                    if majority_label == parent_label:
                        correct_patients += 1

                    # Store patient-wise predictions
                    patient_predictions[os.path.basename(patient_folder)] = {
                        'true_label': parent_label,
                        'predicted_label': majority_label
                    }

                total_patients += 1

            # Print progress after processing each parent folder
            print(f'Processed parent folder: {parent_folder}, Accuracy so far: {correct_patients / total_patients:.4f}')

    patient_level_accuracy = correct_patients / total_patients if total_patients > 0 else 0
    print(f"Correct patients: {correct_patients}")
    print(f"Total patients: {total_patients}")
    print(f'Patient-level accuracy: {patient_level_accuracy:.4f}')

    # Write the results to a text file
    with open('/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/patient_accuracy.txt', 'a') as file:
        file.write(f"Correct patients: {correct_patients}\n")
        file.write(f"Total patients: {total_patients}\n")
        file.write(f'Patient-level accuracy: {patient_level_accuracy:.4f}\n')

    # Save patient-wise predictions to CSV
    df = pd.DataFrame.from_dict(patient_predictions, orient='index')
    df.to_csv('/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/patient_predictions.csv')

# Main function
def main():
    # Paths to your saved model weights (.pth file) and dataset
    model_weights_path = '/raid/student/2021/ai21btech11005/vipcup/classification_from_old_server/classification_models/models/resnet18_single_channel_model.pth'
    dataset_path = '/raid/student/2021/ai21btech11005/vipcup/ICIP training data'

    # Load the model
    print('Loading model...')
    # model = EfficientNet.from_pretrained('efficientnet-b0')
    # model = models.resnet18(pretrained=False)
    model = models.alexnet(pretrained=False)
    
    print('Model loaded successfully')
    # model = convert_to_single_channel(model)
    # model = modify_resnet_for_single_channel(model)
    model = modify_alexnet_for_single_channel(model)
    print('Model converted to single-channel successfully')
    model.load_state_dict(torch.load(model_weights_path))
    print('Model weights loaded successfully')

    model.cuda()

    # Evaluate patient-level accuracy
    print('Evaluating patient-level accuracy...')
    evaluate_patient_level_accuracy(model, dataset_path)
    print('Patient-level accuracy evaluation complete')

if __name__ == '__main__':
    main()
