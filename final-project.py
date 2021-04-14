from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

if __name__ == '__main__':
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    train_loader = get_train_loader('standard', train_data, batch_size=16)

    grouper = CombinatorialGrouper(dataset, ['location'])

