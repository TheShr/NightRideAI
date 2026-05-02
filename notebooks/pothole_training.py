# NightRide AI - Model Training and Evaluation
# Local script for training a pothole model using the repository dataset.

import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image


class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        root_dir = Path(root_dir)
        if root_dir.is_dir() and root_dir.name != 'images' and (root_dir / 'images').is_dir():
            root_dir = root_dir / 'images'

        if not root_dir.exists():
            raise ValueError(f'Dataset directory not found: {root_dir}')

        self.image_paths = sorted(
            p for p in root_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        )
        if not self.image_paths:
            raise ValueError(f'No images found in {root_dir}')

        self.labels = [0 if 'normal' in str(path).lower() else 1 for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total if total > 0 else 0.0
        print(f'Epoch {epoch}/{num_epochs}: Loss: {epoch_loss / total:.4f}, Acc: {train_acc:.2f}%')

    return model


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent if script_path.parent.name == 'notebooks' else script_path.parent
    default_root = repo_root / 'datasets' / 'potholes'
    data_root = Path(os.environ.get('DATA_ROOT', default_root))

    train_root = data_root / 'train' / 'images'
    val_root = data_root / 'val' / 'images'

    print(f'Using data root: {data_root}')
    print(f'Train images: {train_root}')
    print(f'Val images:   {val_root}')

    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError(
            f'Dataset not found. Expected:\n'
            f'  {train_root}\n'
            f'  {val_root}\n'
            f'If your dataset is elsewhere, set DATA_ROOT to the parent folder containing potholes.'
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = PotholeDataset(train_root, transform=transform)
    val_dataset = PotholeDataset(val_root, transform=transform)

    print(f'train samples: {len(train_dataset)}')
    print(f'val samples:   {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)

    trained_model = train_model(model, train_loader, val_loader, num_epochs=10, device=device)

    save_path = repo_root / 'backend' / 'models' / 'pothole_efficientnet.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), save_path)
    print(f'Saved trained model to {save_path}')


if __name__ == '__main__':
    main()

