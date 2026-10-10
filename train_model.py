import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "data",
    "raw",
    "color"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("=" * 60)
print("CROP AI PROJECT")
print("=" * 60)

print("\nProject folder:")
print(PROJECT_DIR)

print("\nDataset folder:")
print(DATA_DIR)

if not os.path.exists(DATA_DIR):
    print("\nERROR: Dataset folder not found!")
    print("Python is looking here:")
    print(DATA_DIR)
    raise FileNotFoundError(DATA_DIR)

print("\nDataset folder found successfully!")


# ============================================================
# 3. CLASSES
# ============================================================

wanted_classes = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",

    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",

    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight"
]


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

full_dataset = datasets.ImageFolder(
    DATA_DIR
)

print("\nTotal classes found:")
print(len(full_dataset.classes))

print("\nTotal images found:")
print(len(full_dataset.samples))


# ============================================================
# 6. SHOW CLASSES
# ============================================================

print("\nAvailable classes:")

for class_name in full_dataset.classes:
    print(" -", class_name)


# ============================================================
# 7. SELECT REQUIRED CLASSES
# ============================================================

print("\n" + "=" * 60)
print("SELECTING REQUIRED 11 CLASSES")
print("=" * 60)

selected_samples = []

for image_path, class_index in full_dataset.samples:

    original_class = full_dataset.classes[class_index]

    if original_class in wanted_classes:

        new_class_index = wanted_classes.index(
            original_class
        )

        selected_samples.append(
            (
                image_path,
                new_class_index
            )
        )


print("\nSelected classes:")

for i, class_name in enumerate(wanted_classes):
    print(i, "-", class_name)

print("\nSelected images:")
print(len(selected_samples))


# ============================================================
# 8. CUSTOM DATASET
# ============================================================

class SelectedDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


dataset = SelectedDataset(
    selected_samples,
    transform=train_transform
)


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("\n" + "=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print("\nTraining images:")
print(len(train_dataset))

print("\nValidation images:")
print(len(val_dataset))


# ============================================================
# 10. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 11. DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("\n" + "=" * 60)
print("DEVICE")
print("=" * 60)

print("\nUsing device:")
print(device)


# ============================================================
# 12. LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(
    weights=weights
)


# ============================================================
# 13. FREEZE FEATURES
# ============================================================

for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# 14. CHANGE CLASSIFIER
# ============================================================

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(wanted_classes)
)

model = model.to(device)


# ============================================================
# 15. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier[1].parameters(),
    lr=0.001
)


# ============================================================
# 16. TRAINING
# ============================================================

EPOCHS = 5

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # ---------------- TRAIN ----------------

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )


    # ---------------- VALIDATION ----------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Loss: {running_loss / len(train_loader):.4f}"
    )

    print(
        f"Training Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# ============================================================
# 17. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "crop_disease_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# 18. SAVE CLASS NAMES
# ============================================================

class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(
    class_names_path,
    "w"
) as file:

    json.dump(
        wanted_classes,
        file,
        indent=4
    )


# ============================================================
# 19. COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModel saved at:")
print(model_path)

print("\nClass names saved at:")
print(class_names_path)

print("\nYour crop disease image-analysis model is ready!")