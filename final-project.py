import os

import analyze_dataset
import dataloading
import models

import torch
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms

def adjust_ys(dataset, num_unique):
    """Adjust labels so that they start at 0.
    
    CURRENTLY UNUSED

    Parameters:
    dataset: WILDSDataset object. The dataset we're modifying. Assumes
        that dataste.y_array consists only of *num_unique* non-negative
        integers.
    num_unique: int. The number of unique labels.

    Returns: WILDSDataset object. The transformed dataset.
    """
    print(torch.unique(dataset.y_array, sorted=True))
    for new_val, y_val in enumerate(torch.unique(dataset.dataset._y_array, sorted=True)):
        # TODO: Make sure that modifying _y_array doesn't violate anything.
        print(y_val)
        print(dataset.dataset.y_array[dataset.dataset.y_array == y_val.item()])
        dataset.dataset._y_array[dataset.dataset._y_array == y_val.item()] = new_val
    
    print(dataset.y_array)


if __name__ == '__main__':

    # Some torch set up
    torch.manual_seed(1234)
    # Default device + GPU random seed setting
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(1234)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        device = torch.device("cpu")

    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')

    # Partition the locations so that no two labels share a location.
    analyze_dataset.create_location_array(train_data, "./location_counts.csv")
    location_lists = analyze_dataset.greedy_location_split("./location_counts.csv", 2000)

    # Create a list of pairs of labels and locations.
    label_ids = [1, 2, 4, 146]
    pair_list = []
    for label, locations in zip(label_ids, location_lists):
        for loc_id in locations:
            pair_list.append((label, loc_id))

    # Modify the dataset to only include indices satisfying these criteria.
    partition_set = dataloading.collect_location_label_pairs(dataset, pair_list, 'train')
    test_pair_list = [(1, 'all'), (2, 'all'), (4, 'all'), (146, 'all')]
    partition_test_set = dataloading.collect_location_label_pairs(dataset, test_pair_list, 'test')
    # partition_set_2 = dataloading.train_collect_location_label_pairs(train_data, pair_list + [(2, 48)])
    # print((partition_set.metadata_array == partition_set_2.metadata_array).all())

    # TODO: Adjust the labels so that they're in the 0 - 3 range?
    # adjust_ys(partition_set, 5)

    partition_loader = get_train_loader('standard', partition_set, batch_size=80)
    eval_loader = get_eval_loader('standard', partition_test_set, batch_size=16)
        
    # partition_set = dataloading.collect_location_label_pairs(train_data, [(idx, idx) for idx in range(5)])
    # partition_loader = get_train_loader('standard', partition_set, batch_size=16)

    model = models.load_modified_pre_trained('resnet18', 4)

    # Place model on device
    model = model.to(device)

    models.fine_tune_model(model, partition_loader, eval_loader, "./tested_model", device, num_epochs=10)

    print(partition_set.metadata_array.shape)
    print(train_data.metadata_array.shape)

    # Set up the testing data.
    #train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    #train_loader = get_train_loader('standard', train_data, batch_size=16)

    #grouper = CombinatorialGrouper(dataset, ['location'])
