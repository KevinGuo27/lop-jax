import torch
import pickle
import torchvision
import torchvision.transforms as transforms

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)



def cifar100():

    transform = transforms.Compose([transforms.ToTensor()])

    train_batch_size = 50000
    test_batch_size = 10000

    train_dataset = torchvision.datasets.CIFAR100(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root="data", train=False, transform=transform
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=True
    )

    for i, (images, labels) in enumerate(train_loader):
        images = images.flatten(start_dim=1)
        labels = labels

    x = images
    y = labels

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test

    x_test = images_test
    y_test = labels_test

    with open('data/cifar100_', 'wb+') as f:
        pickle.dump([x, y, x_test, y_test], f)

    return x, y, x_test, y_test


def get_cifar100(type='reg'):
    if type == 'reg':
        data_file = 'data/cifar100_'
        with open(data_file, 'rb+') as f:
            x, y, x_test, y_test = pickle.load(f)
    return x, y, x_test, y_test


if __name__ == '__main__':
    """
    Generates all the required data
    """
    cifar100()