import analyze_dataset
import dataloading
import models

import torch
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

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
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')

    # Partition the locations so that no two labels share a location.
    analyze_dataset.create_location_array(train_data, "./location_counts.csv")
    location_lists = analyze_dataset.greedy_location_split("./location_counts.csv", 20)

    # Create a list of pairs of labels and locations.
    label_ids = [1, 2, 4, 146]
    pair_list = []
    for label, locations in zip(label_ids, location_lists):
        for loc_id in locations:
            pair_list.append((label, loc_id))

    # Modify the dataset to only include indices satisfying these criteria.
    partition_set = dataloading.collect_location_label_pairs(dataset, pair_list, 'train')
    # partition_set_2 = dataloading.train_collect_location_label_pairs(train_data, pair_list + [(2, 48)])
    # print((partition_set.metadata_array == partition_set_2.metadata_array).all())

    # TODO: Adjust the labels so that they're in the 0 - 3 range?
    # adjust_ys(partition_set, 5)

    partition_loader = get_train_loader('standard', partition_set, batch_size=16)
        
    # partition_set = dataloading.collect_location_label_pairs(train_data, [(idx, idx) for idx in range(5)])
    # partition_loader = get_train_loader('standard', partition_set, batch_size=16)

    model = models.load_modified_pre_trained('resnet18', 4)

    models.fine_tune_model(model, partition_loader, "./hmm_model", num_epochs=10)

    print(partition_set.metadata_array.shape)
    print(train_data.metadata_array.shape)

    # Set up the testing data.
    #train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    #train_loader = get_train_loader('standard', train_data, batch_size=16)

    #grouper = CombinatorialGrouper(dataset, ['location'])
