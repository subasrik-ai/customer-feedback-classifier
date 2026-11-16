
# ============================================================
# TASK 2: PYTORCH TRAINING PIPELINE
# PyTorch-only synthetic binary classification dataset
# ============================================================

import os
import csv
import random
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# STEP 1: REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# STEP 2: GENERATE A SYNTHETIC DATASET
# ============================================================

# This is temporary practice data, NOT real breast cancer data.
# Each sample has 30 features and a binary label.

def generate_sample(label):

    features = []

    for i in range(30):

        if i < 5:
            # First five features contain a class signal
            mean = 1.0 if label == 1 else -1.0
        else:
            mean = 0.0

        features.append(random.gauss(mean, 1.0))

    return features, label


all_samples = []

for _ in range(250):
    all_samples.append(generate_sample(0))
    all_samples.append(generate_sample(1))

random.shuffle(all_samples)

split_index = 400

train_samples = all_samples[:split_index]
val_samples = all_samples[split_index:]

print("\nDataset created!")
print("Training samples:", len(train_samples))
print("Validation samples:", len(val_samples))


# ============================================================
# STEP 3: SCALE FEATURES WITHOUT NUMPY
# ============================================================

# Calculate means using training data only.

feature_count = 30

means = []

for column in range(feature_count):

    column_values = [
        sample[0][column]
        for sample in train_samples
    ]

    means.append(
        sum(column_values) / len(column_values)
    )


# Calculate standard deviations.

stds = []

for column in range(feature_count):

    column_values = [
        sample[0][column]
        for sample in train_samples
    ]

    variance = sum(
        (value - means[column]) ** 2
        for value in column_values
    ) / len(column_values)

    std = variance ** 0.5

    stds.append(std if std > 0 else 1.0)


# Apply scaling to both datasets.

def scale_samples(samples):

    scaled = []

    for features, label in samples:

        scaled_features = [
            (features[i] - means[i]) / stds[i]
            for i in range(feature_count)
        ]

        scaled.append((scaled_features, label))

    return scaled


train_scaled = scale_samples(train_samples)
val_scaled = scale_samples(val_samples)


# ============================================================
# STEP 4: SAVE DATA AS CSV FILES
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def save_csv(filename, samples):

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        for features, label in samples:

            writer.writerow(features + [label])


save_csv("data/train.csv", train_scaled)
save_csv("data/validation.csv", val_scaled)

print("\nCSV files saved!")


# ============================================================
# STEP 5: CREATE CUSTOM CSV DATASET
# ============================================================

class CsvDataset(Dataset):

    def __init__(self, csv_file):

        self.features = []
        self.labels = []

        with open(csv_file, "r") as file:

            reader = csv.reader(file)

            for row in reader:

                values = [float(value) for value in row]

                self.features.append(values[:-1])
                self.labels.append([values[-1]])

        self.features = torch.tensor(
            self.features,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index]
        )


train_dataset = CsvDataset("data/train.csv")
val_dataset = CsvDataset("data/validation.csv")

print("\nTraining dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))


# ============================================================
# STEP 6: CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("\nDataLoaders created!")


# ============================================================
# STEP 7: BUILD MLP NEURAL NETWORK
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(30, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nParams:", parameter_count)


# ============================================================
# STEP 8: LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# STEP 9: VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            total_loss += loss.item() * labels.size(0)

    accuracy = correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


# ============================================================
# STEP 10: TRAINING LOOP WITH EARLY STOPPING
# ============================================================

epochs = 40
patience = 8

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_epoch = 0

epochs_without_improvement = 0

best_model_path = "models/breast_cancer_mlp.pth"

training_losses = []
validation_accuracies = []

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    total_samples = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        # 1. Clear old gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(features)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    train_loss = running_loss / total_samples

    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    training_losses.append(train_loss)
    validation_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    # Early stopping monitors validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:

        print("\nEarly stopping triggered!")
        break


print("\nTraining completed!")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f} at epoch {best_epoch}"
)


# ============================================================
# STEP 11: LOAD BEST MODEL INTO A NEW MODEL
# ============================================================

new_model = MLP().to(device)

saved_state = torch.load(
    best_model_path,
    map_location=device
)

new_model.load_state_dict(saved_state)
new_model.eval()

print("\nSaved model loaded successfully!")


# ============================================================
# STEP 12: VERIFY IDENTICAL ACCURACY
# ============================================================

# Evaluate the best checkpoint in both models.

best_model = MLP().to(device)
best_model.load_state_dict(saved_state)
best_model.eval()

_, best_accuracy = evaluate(best_model, val_loader)

_, reloaded_accuracy = evaluate(new_model, val_loader)

print(
    f"\nBest model accuracy: {best_accuracy:.4f}"
)

print(
    f"Reloaded model accuracy: {reloaded_accuracy:.4f}"
)

print(
    "Accuracy identical:",
    best_accuracy == reloaded_accuracy
)


# ============================================================
# STEP 13: FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Parameters:", parameter_count)
print("Best epoch:", best_epoch)

print(
    f"Best validation accuracy: {best_accuracy:.4f}"
)

print(
    f"Reloaded model accuracy: {reloaded_accuracy:.4f}"
)

print("Saved model:", best_model_path)


