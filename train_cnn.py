import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

def get_loaders(train_dir: str, val_dir: str, test_dir: str, img_size: int, batch_size: int, num_workers: int):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = get_loaders(
        args.train_dir, args.val_dir, args.test_dir, args.img_size, args.batch_size, args.num_workers
    )

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    # Transfer learning: ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    best_path = Path(args.out_dir) / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            pbar.set_postfix(loss=running_loss / max(total, 1), acc=correct / max(total, 1))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size,
            }, best_path)
            print(f"Saved best model to: {best_path} (val_acc={best_val_acc:.4f})")

    # Final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"\n Final Test: loss={test_loss:.4f} acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
