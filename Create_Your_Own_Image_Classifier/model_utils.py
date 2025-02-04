import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

def create_model(arch='vgg16', hidden_units=512, learning_rate=0.003):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
        model.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_units)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported architecture: Choose 'vgg16' or 'resnet18'")

    criterion = nn.NLLLoss()
    
    return model, criterion, optimizer

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def save_checkpoint(model, save_dir, arch, class_to_idx):
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

def load_checkpoint(filepath):
    """Load model from checkpoint."""
    checkpoint = torch.load(filepath)
    model, _, _ = create_model(checkpoint['arch'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
