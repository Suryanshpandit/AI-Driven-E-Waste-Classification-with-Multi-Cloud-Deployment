import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print("Predicted class:", classes[pred])

if __name__ == "__main__":
    main()
