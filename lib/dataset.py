import torch
import torchvision
import torchvision.transforms as transforms

class dataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        self.trainloader_pool = {}
        self.testloader_pool = {}
        if not (self.dataset in set(['cifar10', 'cifar100', 'svhn'])):
            print('dataset {} is not supported.'.format(self.dataset))
            raise RuntimeError

    def trainloader(self, batch_size=128, data_augment=True):
        key = '{}_{}'.format(batch_size, data_augment)
        if self.trainloader_pool.get(key):
            return self.trainloader_pool.get(key)

        transform_list = []
        if data_augment:
            transform_list.append(transforms.RandomCrop(32, padding=4))
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)

        if self.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose(transform_list))
        elif self.dataset == 'cifar100':
            train_dataset = torchvision.datasets.CIFAR100(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose(transform_list))
        elif self.dataset == 'svhn':
            train_dataset = torchvision.datasets.SVHN(
                root='./data',
                split='train',
                download=True,
                transform=transforms.Compose(transform_list))
        else:
            raise RuntimeError

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.trainloader_pool[key] = train_loader
        return train_loader

    def testloader(self, batch_size=100):
        key = str(batch_size)
        if self.testloader_pool.get(key):
            return self.testloader_pool.get(key)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(self.normalize)

        if self.dataset == 'cifar10':
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=transforms.Compose(transform_list)
            )
        elif self.dataset == 'cifar100':
            test_dataset = torchvision.datasets.CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=transforms.Compose(transform_list)
            )
        elif self.dataset == 'svhn':
            test_dataset = torchvision.datasets.SVHN(
                root='./data',
                split='test',
                download=True,
                transform=transforms.Compose(transform_list))
        else:
            raise RuntimeError

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.testloader_pool[key] = test_loader
        return test_loader

    def num_classes(self):
        if self.dataset == 'cifar10':
            return 10
        elif self.dataset == 'cifar100':
            return 100
        elif self.dataset == 'svhn':
            return 10
        else:
            raise RuntimeError
