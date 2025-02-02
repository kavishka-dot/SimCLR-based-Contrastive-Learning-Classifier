import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as DataLoader

class SimCLRTransform:
    '''Transforms for SimCLR'''
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)
    
def get_Dataloader():
    '''Load CIFAR-10 dataset'''
    # Download CIFAR-10 Dataset
    batch_size = 256

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=SimCLRTransform(), download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=SimCLRTransform(), download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return train_loader, test_loader