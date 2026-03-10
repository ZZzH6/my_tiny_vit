import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64, num_workers=4, data_dir='./data'):
    """构建训练和验证数据加载器。"""
    # 移除了极其耗时的 Resize(224, 224)，回归 CIFAR 原生 32x32 大小
    # 使用标准 CIFAR 数据增强：RandomCrop(32, padding=4)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       
        transforms.RandomHorizontalFlip(),   
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), 
        transforms.ToTensor(),               
        transforms.Normalize(                
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        ),
        transforms.RandomErasing(p=0.25),  # 随机擦除增强，进一步防过拟合
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    print("正在加载 CIFAR-100 数据集...")
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset
