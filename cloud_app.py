from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

# Load model
device = "cpu"
ckpt = torch.load("outputs/best_model.pt", map_location=device)

classes = ckpt["classes"]
img_size = ckpt["img_size"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    return jsonify({
        "prediction": classes[pred]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
