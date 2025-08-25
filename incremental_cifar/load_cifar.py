import torch
import pickle
import torchvision
import torchvision.transforms as transforms
import numpy as np

def cifar100():
    """
    Loads CIFAR-100 and saves it in a pickle file
    """
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=10000, shuffle=True, num_workers=2
    )

    # grab the one big batch
    x_train_t, y_train_t = next(iter(train_loader))
    x_test_t,  y_test_t  = next(iter(test_loader))

    # to numpy
    x_train_np = x_train_t.numpy()
    y_train_np = y_train_t.numpy()
    x_test_np  = x_test_t.numpy()
    y_test_np  = y_test_t.numpy()

    # make labels one-hot
    num_classes = 100
    y_train_oh_np = np.eye(num_classes)[y_train_np]
    y_test_oh_np  = np.eye(num_classes)[y_test_np]

    # save as a dict for clarity
    with open('./data/cifar100-onehot.pkl', 'wb') as f:
        pickle.dump({
            'x_train': x_train_np,
            'y_train': y_train_oh_np,
            'x_test':  x_test_np,
            'y_test':  y_test_oh_np
        }, f)


if __name__ == '__main__':
    """
    Generates all the required data
    """
    cifar100()