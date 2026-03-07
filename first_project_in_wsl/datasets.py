import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, num_workers=4, data_dir='./data'):
    """构建训练和验证数据加载器。"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),       
        transforms.RandomHorizontalFlip(),   
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), 
        transforms.ToTensor(),               
        transforms.Normalize(                
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("正在加载 CIFAR-100 数据集...")
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset
