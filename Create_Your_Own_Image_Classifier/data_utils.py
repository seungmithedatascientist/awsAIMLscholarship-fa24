from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir):
    """Load training, validation, and test datasets."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx
