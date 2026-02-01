"""pytorchexample: A Flower / PyTorch app."""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
   
    """Improved CNN for CIFAR-like images"""

    def __init__(self):
        super().__init__()

        # -------- Feature extractor --------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # -------- Classifier --------
        # CIFAR: 32x32 → pool → 16x16 → pool → 8x8 → pool → 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)




class CIFARFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append(
                    (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label



def load_local_image_data(client_id: int, batch_size: int):
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(),T.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))])
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", f"client_{client_id}"))

    dataset = CIFARFolderDataset(root, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size)

    return trainloader, valloader



def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss




def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def class_wise_accuracy(net, dataloader, device, num_classes=10):
    net.eval()
    correct = [0] * num_classes
    total = [0] * num_classes

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                if preds[i].item() == label:
                    correct[label] += 1

    return {
        f"class_{c}_acc": (correct[c] / total[c] if total[c] > 0 else 0.0)
        for c in range(num_classes)
    }