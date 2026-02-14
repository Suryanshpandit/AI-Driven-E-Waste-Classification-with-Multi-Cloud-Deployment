import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "outputs/best_model.pt"
TEST_DIR = "D:/major project/ewaste_project/E-waste dataset/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
ckpt = torch.load(MODEL_PATH, map_location=device)
classes = ckpt["classes"]
img_size = ckpt["img_size"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# Transforms
tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# Dataset
test_ds = datasets.ImageFolder(TEST_DIR, transform=tfm)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - E-Waste Classification")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))
