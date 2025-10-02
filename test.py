import torch
import torchvision
import os
from torchvision import models, datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse  # Import argparse

# Check the current directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# Define test data folder
test_set_path = './testdata/'

# Check if the test data folder exists
if os.path.exists(test_set_path):
    print(f"Test data folder '{test_set_path}' exists.")
else:
    print(f"Test data folder '{test_set_path}' does not exist.")
    exit()  # Exit if the folder does not exist

# Check if CUDA, MPS, or CPU is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device used: {device}")

# Define transformations
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set parameters
batch_size_test = 64

# Initialize dataset and dataloader
test_dataset = datasets.ImageFolder(root=test_set_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

# Define the ResNet18 model
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.38359):
        super(ResNet18Model, self).__init__()
        self.cnn_model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.cnn_model(x)

# Function to evaluate the model on the test set
def model_evaluation(model, test_loader, device):
    model.eval()
    correct_predictions = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Store all true labels and predictions for later analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Print progress
            print(f'Processing batch {batch_index + 1}/{len(test_loader)}: {labels.size(0)} images processed.')

    # Calculate overall accuracy
    accuracy = correct_predictions / total
    print(f'Accuracy on unseen data: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, zero_division=0))

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=np.unique(all_labels))

    return accuracy, conf_matrix

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('model_PATH', type=str, help='Path to the trained model file (e.g., model.pth)')
    args = parser.parse_args()

    # Load the model
    model = ResNet18Model(num_classes=3)
    model.load_state_dict(torch.load(args.model_PATH, map_location=device))
    model.eval()
    model = model.to(device)

    # Run the evaluation
    model_evaluation(model, test_loader, device)
