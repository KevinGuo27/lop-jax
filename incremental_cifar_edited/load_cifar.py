import torch
import pickle
import torchvision
import torchvision.transforms as transforms

def cifar100():
    """
    Loads CIFAR-100 and saves it in a pickle file
    """
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
        transforms.ToTensor(),  # reshape to (C x H x W)
        transforms.Normalize(mean=mean, std=std),  # center by mean and divide by std
        transforms.RandomCrop(size=32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0,15))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000,
                                              shuffle=False, num_workers=2)

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)  # center by mean and divide by std
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False, num_workers=2)

    for data in trainloader:
        all_x_train, all_y_train = data
    for data in testloader:
        all_x_test, all_y_test = data

    with open('./data/cifar100.pkl', 'wb') as f:
        pickle.dump((all_x_train.numpy(), all_y_train.numpy(), all_x_test.numpy(), all_y_test.numpy()), f)


if __name__ == '__main__':
    """
    Generates all the required data
    """
    cifar100()