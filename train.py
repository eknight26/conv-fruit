import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import itertools
from torchvision.models import ResNet34_Weights, VGG19_Weights
import time
from tqdm import tqdm

%matplotlib inline

# Preprocess the images (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((64, 64)),                             # Resize images to 64x64
    transforms.ToTensor()                                    # Convert images to tensor
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=training_set_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)   

# Initialize variables for mean and std
mean = torch.zeros(3)                                        # 3 channels (RGB)
stdv = torch.zeros(3)
total_images = 0

# Calculate mean and std per channel
for images, _ in loader:
    batch_size = images.size(0)
    mean += images.mean(dim=[0, 2, 3]) * batch_size          # Mean for each channel
    stdv += images.std(dim=[0, 2, 3]) * batch_size           # Std for each channel
    total_images += batch_size

# Final mean and std calculations
mean /= total_images
stdv /= total_images

print("Mean:", mean)
print("Standard Deviation:", stdv)



#########################
#### Simple MLP Model ###
#########################

# Define image dimensions
image_size = 64                           # Resizing images to 64x64
input_size = image_size * image_size * 3  # Flattened size for 64x64 RGB images

# Number of classes
num_classes = 3  

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        # Define fully connected layers with an additional fourth hidden layer
        self.fc1 = nn.Linear(input_size, 512)    # First hidden layer
        self.fc2 = nn.Linear(512, 256)           # Second hidden layer
        self.fc3 = nn.Linear(256, 128)           # Third hidden layer
        self.fc4 = nn.Linear(128, 64)            # Fourth hidden layer
        self.fc5 = nn.Linear(64, num_classes)    # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)                # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))                  # Apply fourth layer
        x = self.fc5(x)                          # Output layer (no activation if using CrossEntropyLoss)
        return x

# Initialize the MLP model
mlp_model = SimpleMLP(input_size=input_size, num_classes=num_classes)

# Move model to appropriate device
mlp_model.to(device)

# Function to initialize the model
def initialise_model():
    return SimpleMLP(input_size=input_size, num_classes=num_classes).to(device)


class EarlyStoppingLoss:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience                            # Number of epochs with no improvement after which training will be stopped
        self.min_delta = min_delta                          # Minimum improvement threshold
        self.best_loss = float('inf')                       # Initialize with a very high loss
        self.counter = 0                                    # Counter to track epochs with no improvement

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:      # Check if the current loss is an improvement
            self.best_loss = val_loss                       # Update the best loss
            self.counter = 0                                # Reset the counter
            return False                                    # No early stopping

        # If no improvement, increase the counter
        self.counter += 1

        # Stop training if the patience threshold is reached
        if self.counter >= self.patience:
            return True                                     # Early stopping triggered

        return False                                        # Continue training


# Preprocess the images
mlp_train_transforms = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),                  # Expanded scale
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),                           
    transforms.Normalize(mean, stdv)
])

val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, stdv)
])

# Define values
epochs = 30
patience = 5
batch_size = 128  
mlp_early_stopping = EarlyStoppingLoss(patience=patience, min_delta=0.001)  # Initialise early stopping

# Initialise model, optimizer, and criterion
mlp_model = initialise_model().to(device)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.0001, weight_decay=1e-3)
mlp_criterion = nn.CrossEntropyLoss()

# Define the number of splits for StratifiedKFold
mlp_stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=309)

# Extract labels from the dataset for stratified splitting
mlp_targets = [dataset[i][1] for i in range(len(dataset))]

# Get the first split of indices
mlp_train_indices, mlp_val_indices = next(mlp_stratified_kfold.split(np.arange(len(dataset)), mlp_targets))

# Subset datasets for stratified train and validation sets
mlp_train_data = Subset(dataset, mlp_train_indices)
mlp_val_data = Subset(dataset, mlp_val_indices)

# Apply transformations
mlp_train_data.dataset.transform = mlp_train_transforms
mlp_val_data.dataset.transform = val_transforms

# Dataloaders
mlp_train_loader = DataLoader(mlp_train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
mlp_val_loader = DataLoader(mlp_val_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialise lists to track losses and accuracies
mlp_train_losses = []
mlp_val_losses = []
mlp_train_accuracies = []
mlp_val_accuracies = []

# Training loop
for mlp_epoch in range(epochs):
    mlp_model.train()
    running_loss, mlp_correct, mlp_total = 0.0, 0, 0

    # Wrap the mlp_train_loader with tqdm for progress tracking
    with tqdm(mlp_train_loader, desc=f'Epoch {mlp_epoch + 1}/{epochs}', unit='batch') as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            mlp_optimizer.zero_grad()

            outputs = mlp_model(images)
            loss = mlp_criterion(outputs, labels)
            loss.backward()
            mlp_optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            mlp_total += labels.size(0)
            mlp_correct += (predicted == labels).sum().item()

            # Update tqdm description with loss and accuracy
            tepoch.set_postfix(loss=running_loss / (tepoch.n + 1), accuracy=mlp_correct / mlp_total)

    mlp_train_loss = running_loss / len(mlp_train_loader)
    mlp_train_losses.append(mlp_train_loss)
    mlp_train_accuracy = mlp_correct / mlp_total
    mlp_train_accuracies.append(mlp_train_accuracy)
    print(f'Epoch {mlp_epoch + 1}/{epochs}, Loss: {mlp_train_loss:.4f}, Accuracy: {mlp_train_accuracy * 100:.2f}%')

    # Validation phase
    mlp_model.eval()
    mlp_val_loss, mlp_val_correct, mlp_val_total = 0.0, 0, 0
    all_labels, all_predictions = [], []

    # Wrap the mlp_val_loader with tqdm for progress tracking
    with tqdm(mlp_val_loader, desc='Validation', unit='batch') as vloader:
        with torch.no_grad():
            for val_images, val_labels in vloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = mlp_model(val_images)
                mlp_val_loss += mlp_criterion(val_outputs, val_labels).item()

                _, val_predicted = torch.max(val_outputs, 1)
                mlp_val_total += val_labels.size(0)
                mlp_val_correct += (val_predicted == val_labels).sum().item()

                all_labels.extend(val_labels.cpu().numpy())
                all_predictions.extend(val_predicted.cpu().numpy())

                # Update tqdm description for validation
                vloader.set_postfix(val_loss=mlp_val_loss / (vloader.n + 1), val_accuracy=mlp_val_correct / mlp_val_total)

    mlp_avg_val_loss = mlp_val_loss / len(mlp_val_loader)
    mlp_val_losses.append(mlp_avg_val_loss)
    mlp_val_accuracy = mlp_val_correct / mlp_val_total
    mlp_val_accuracies.append(mlp_val_accuracy)
    print(f'Validation Loss: {mlp_avg_val_loss:.4f}, Validation Accuracy: {mlp_val_accuracy * 100:.2f}%')

    # Early stopping based on validation loss
    if mlp_early_stopping.step(mlp_avg_val_loss):
        print("Early stopping triggered based on validation loss")
        break

# Save the final model
path = './mlp_model.pth'
torch.save(mlp_model.state_dict(), path)
print(f'Model saved at {path}')

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, zero_division=0))

# Assuming your class labels are as follows:
mlp_class_labels = ['cherry', 'strawberry', 'tomato']

# Confusion Matrix
mlp_cm = confusion_matrix(all_labels, all_predictions)
mlp_cm_df = pd.DataFrame(mlp_cm, index=mlp_class_labels, columns=mlp_class_labels)


epochs = len(mlp_train_losses)

# Create a DataFrame
data = {
    'Epoch': list(range(1, epochs + 1)),
    'Training Loss': mlp_train_losses,
    'Validation Loss': mlp_val_losses,
    'Training Accuracy': mlp_train_accuracies,
    'Validation Accuracy': mlp_val_accuracies
}

df = pd.DataFrame(data)

# Melt the DataFrame to long format for easier plotting
df_melted = df.melt(id_vars='Epoch', value_vars=['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'],
                    var_name='Metric', value_name='Value')

# Create the plot
plt.figure(figsize=(15, 7))
sns.lineplot(data=df_melted, x='Epoch', y='Value', hue='Metric', marker='o')

# Customize the plot
plt.title('', fontsize=16)  
plt.xlabel('Epochs', fontsize=16)  
plt.ylabel('Accuracy / Loss', fontsize=16)  
plt.grid()
plt.legend(title='', fontsize=12, title_fontsize='13')  
plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')

plt.tick_params(axis='both', labelsize=12)  

plt.show()


print(f"Mean MLP Training Loss: {torch.mean(torch.tensor(mlp_train_losses)):.2f}")
print(f"Mean MLP Validation Loss: {torch.mean(torch.tensor(mlp_val_losses)):.2f}")



#############
#### CNN ####
#############

# Training functions
def train_one_epoch(model, loader, criterion, optimizer_instance, loss_name):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training batches"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_instance.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) if loss_name == 'cross_entropy' else criterion(nn.LogSoftmax(dim=1)(outputs), labels)
        loss.backward()
        optimizer_instance.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return running_loss / len(loader), accuracy


def validate(model, loader, criterion, loss_name):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) if loss_name == 'cross_entropy' else criterion(nn.LogSoftmax(dim=1)(outputs), labels)
            val_loss += loss.item()

            _, val_predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(val_predicted.cpu().numpy())
            val_correct += (val_predicted == labels).sum().item()
            val_total += labels.size(0)
    
    accuracy = val_correct / val_total
    return val_loss / len(loader), accuracy, all_labels, all_predictions


### Hyperparameter Tuning using OPTUNA ###
# Define the ResNet18 model
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=None):
        super(ResNet18Model, self).__init__()
        self.cnn_model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.cnn_model.fc.in_features
        self.cnn_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.cnn_model(x)

import optuna

# Define augmentations
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming you have a dataset structured in a way compatible with ImageFolder
train_data = datasets.ImageFolder(root=training_set_path, transform=train_transforms)

labels = np.array(train_data.targets)

def train_and_validate(trial):
    # Parameters
    num_epochs = 20
    weight_decay = 1e-4
    num_classes = 3
    
    # Define hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])
    loss_name = trial.suggest_categorical("loss_function", ['nll_loss', 'cross_entropy'])
    opt_name = trial.suggest_categorical("optimizer", ['adam', 'rmsprop'])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    # Loss function selection
    criterion = nn.NLLLoss() if loss_name == 'nll_loss' else nn.CrossEntropyLoss()

    # Optimizer selection
    optimizer_class = {
        'adam': lambda params: optim.Adam(params, lr=learning_rate, weight_decay=weight_decay),
        'rmsprop': lambda params: optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
    }

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=3)
    best_val_loss = float("inf")

    for train_index, val_index in skf.split(np.zeros(len(labels)), labels):
        # Create subsets for training and validation
        train_subset = Subset(train_data, train_index)

        # Create DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        # For validation, we need to reapply the val_transforms
        val_data = datasets.ImageFolder(root=training_set_path, transform=val_transforms)      # Reapplying val_transform to avoid data leakage
        val_subset = Subset(val_data, val_index)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Model setup
        cnn_model = ResNet18Model(num_classes=3, dropout_rate=dropout_rate).to(device)

        # Optimizer and scheduler
        optimizer_instance = optimizer_class[opt_name](cnn_model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance, mode='min', factor=0.5, patience=3)

        # Early stopping parameters
        patience, stopping_counter = 3, 0
        fold_best_val_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_train_loss, train_accuracy = train_one_epoch(cnn_model, train_loader, criterion, optimizer_instance, loss_name)
            epoch_val_loss, val_accuracy, all_labels, all_predictions = validate(cnn_model, val_loader, criterion, loss_name)

            if epoch_val_loss < fold_best_val_loss:
                fold_best_val_loss = epoch_val_loss
                stopping_counter = 0  # Reset counter if improvement
            else:
                stopping_counter += 1

            if stopping_counter >= patience:
                print(f"Early stopping on epoch {epoch + 1}")
                break

            scheduler.step(epoch_val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        best_val_loss = min(best_val_loss, fold_best_val_loss)

    return best_val_loss


# Optuna Study
study = optuna.create_study(direction='minimize')
study.optimize(train_and_validate, n_trials=10)  

# Print the best parameters
print("Best hyperparameters: ", study.best_params)
print("Best validation loss: ", study.best_value)


# Cross Validation

# define functions

def train_one_epoch(model, loader, criterion, optimizer_instance):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training batches"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_instance.zero_grad()
        outputs = model(inputs)
        loss = criterion(nn.LogSoftmax(dim=1)(outputs), labels)      
        loss.backward()
        optimizer_instance.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return running_loss / len(loader), accuracy

def validate(model, loader, criterion):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_labels, all_predictions = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(nn.LogSoftmax(dim=1)(outputs), labels)
            val_loss += loss.item()

            _, val_predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(val_predicted.cpu().numpy())
            val_correct += (val_predicted == labels).sum().item()
            val_total += labels.size(0)
    
    accuracy = val_correct / val_total
    return val_loss / len(loader), accuracy, all_labels, all_predictions

# Parameters
num_epochs = 30
num_classes = 3
weight_decay = 1e-4
learning_rate = study.best_params['learning_rate']                        
batch_size = study.best_params['batch_size']
dropout = study.best_params['dropout_rate']

train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the number of splits for StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=309)

# Initialize variables to track metrics across folds
all_train_losses = []
all_val_losses = []
all_accuracies = []

train_data = datasets.ImageFolder(root=training_set_path, transform=train_transforms)
# Extract labels from the dataset for stratified splitting
labels = np.array(train_data.targets)

for fold, (train_indices, val_indices) in enumerate(stratified_kfold.split(np.arange(len(train_data)), labels)):
    print(f"Starting fold {fold + 1}/{stratified_kfold.n_splits}")
    
    # Create subsets for training and validation
    train_subset = Subset(train_data, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
    # For validation, we need to reapply the val_transforms
    val_data = datasets.ImageFolder(root=training_set_path, transform=val_transforms)      # Reapplying val_transform to avoid data leakage
    val_subset = Subset(val_data, val_indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Initialize a new model instance for each fold
    cnn_model = ResNet18Model(num_classes=num_classes, dropout_rate=dropout).to(device)
    
    # Define optimizer, scheduler, and criterion
    optimizer_instance = optim.RMSprop(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance, mode='min', factor=0.5, patience=5)
    criterion = nn.NLLLoss()
    
    # Initialize early stopping for each fold
    early_stopping = EarlyStoppingLoss(patience=5, min_delta=0.001)
    
    # Track losses and accuracy for each fold
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        epoch_train_loss, train_accuracy = train_one_epoch(cnn_model, train_loader, criterion, optimizer_instance)
        epoch_val_loss, val_accuracy, all_labels, all_predictions = validate(cnn_model, val_loader, criterion)
        
        # Append losses for each epoch
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(cnn_model.state_dict(), f"best_model_fold_{fold + 1}.pth")
            print("Model saved as best for fold", fold + 1)
        
        scheduler.step(epoch_val_loss)
        
        # Early stopping
        if early_stopping.step(epoch_val_loss):
            print("Early stopping triggered for fold", fold + 1)
            break
    
    # Collect training/validation losses and accuracies for each fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_accuracies.append(val_accuracy)  # Collect final validation accuracy of the fold

# Calculate and print average accuracy
average_accuracy = np.mean(all_accuracies)
print(f"\nAverage Validation Accuracy across {stratified_kfold.n_splits} folds: {average_accuracy:.2f}%")

# Calculate mean losses and accuracies
mean_train_loss = np.mean([np.mean(train) for train in all_train_losses])
mean_val_loss = np.mean([np.mean(val) for val in all_val_losses])
mean_accuracy = np.mean(all_accuracies)

# Print the mean values
print(f'Mean Training Loss across all folds: {mean_train_loss:.2f}')
print(f'Mean Validation Loss across all folds: {mean_val_loss:.2f}')
print(f'Mean Accuracy across all folds: {mean_accuracy:.2f}%')


study.best_params

# Parameters including obtained from tuning
num_epochs = 30
num_classes = 3
weight_decay = 1e-4
learning_rate = 0.000237                         
batch_size = 64
dropout = 0.38359

# some changes to transformations done to feed network with newish data
train_transforms_alternative = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.2),  
    transforms.RandomRotation(degrees=25),  
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load full dataset with training transformations
full_data = datasets.ImageFolder(root=training_set_path, transform=train_transforms)

# Split data into train and validation sets
train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = torch.utils.data.random_split(full_data, [train_size, val_size])

# Override transform for validation dataset
val_data.dataset.transform = val_transforms

# Create DataLoaders for train and validation sets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# Model setup
cnn_model = ResNet18Model(num_classes=num_classes, dropout_rate=dropout).to(device)

# Define optimizer, scheduler, and criterion
optimizer_instance = optim.RMSprop(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_instance, mode='min', factor=0.5, patience=5)
criterion = nn.NLLLoss()

# Early stopping
early_stopping = EarlyStoppingLoss(patience=10, min_delta=0.001)

# Tracking losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float("inf")

for epoch in range(num_epochs):
    epoch_train_loss, train_accuracy = train_one_epoch(cnn_model, train_loader, criterion, optimizer_instance)
    epoch_val_loss, val_accuracy, all_labels, all_predictions = validate(cnn_model, val_loader, criterion)
    
    # Print epoch metrics
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, '
          f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%')

    # Append losses and accuracies
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Save model if validation loss is best
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(cnn_model.state_dict(), f"best_model_{best_val_loss}_{epoch}.pth")
        print("Model saved as best for current configuration")

    scheduler.step(epoch_val_loss)

    # Early stopping
    if early_stopping.step(epoch_val_loss):
        print("Early stopping triggered.")
        break

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
cm_df = pd.DataFrame(cm, index=['cherry', 'strawberry', 'tomato'], columns=['cherry', 'strawberry', 'tomato'])

plt.figure(figsize=(8, 5))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('Training completed.')


# Plotting losses and accuracies
cnn_epochs = range(1, len(train_losses) + 1)

# Create DataFrame for plotting
metrics_data = pd.DataFrame({
    'Epoch': cnn_epochs,
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'Training Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
})

# Melt the DataFrame for easier plotting
metrics_data_melted = metrics_data.melt(id_vars='Epoch', 
                                         value_vars=['Training Loss', 'Validation Loss', 
                                                     'Training Accuracy', 'Validation Accuracy'],
                                         var_name='Metric', value_name='Value')

# Create the plot
plt.figure(figsize=(15, 7))
sns.lineplot(data=metrics_data_melted, x='Epoch', y='Value', hue='Metric', marker='o')

# Customize the plot
plt.title('', fontsize=16)  
plt.xlabel('Epochs', fontsize=16)  
plt.ylabel('Accuracy / Loss', fontsize=16)  
plt.grid()
plt.legend(title='', fontsize=14, title_fontsize='13')  
plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.tick_params(axis='both', labelsize=12)  

plt.grid()

# Show the plot
plt.show()



