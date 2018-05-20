import torch
import torchvision
import torchvision.transforms as transforms

def dataloader_factory(dataset, batch_size, data_augment=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    transform_list = []
    if data_augment:
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(transform_list))
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(transform_list))
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif dataset == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transforms.Compose(transform_list))
        test_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise ('dataset {} is not supported.'.format(dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    return (train_loader, test_loader)
