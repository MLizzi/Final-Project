"""Methods for loading and customizing datasets."""

import numpy as np
from wilds.datasets.wilds_dataset import WILDSSubset
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

def collect_groups(train_dataset, groups, labels):
    """Create a subset of the data only including these groups and labels.

    Parameters:
    train_dataset: WILDSDataset object. The original training dataset.
    groups: list of int. The locations of interest.
    labels: list of int. The labels of interest.

    Returns: WILDSSubset object.

    """
    # Location is the first piece of metadata in the IWILDS dataset.

    # Determine which indices satisfy both of our criteria, and then create
    # a map for both using the AND operator.
    location_map = np.isin(train_dataset.metadata_array[:,0], groups)
    label_map = np.isin(train_dataset.y_array, labels)

    both_map = np.logical_and(location_map, label_map)
    both_idx = np.where(both_map)[0]

    # Create a WILDSSubset object using the train_dataset and these indices.
    # Include a transform that converts the images to tensors, resizes them,
    # and then normalize them according to the IMAGENET normalization values.
    # More details can be found here: https://pytorch.org/vision/stable/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    composed = transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor(), normalize])

    return WILDSSubset(train_dataset, both_idx, composed)


def collect_location_label_pairs(train_dataset, pairs):
    """Create a subset of the data so that each *label* sample comes from *location*.

    Parameters:
    train_dataset: WILDSDataset object. The original training dataset.
    pairs: list of tuples of ints. Describing pairs of locations and labels. For
        each pair (label, location), OUR dataset will only contain *label* samples
        from *location*. For example, [(1,2), (3,4)] will result in a dataset
        containing only samples of class 1 from location 2, and samples
        of class 3 from location 4.

    Returns: WILDSSubset object.

    """
    # We'll continually update a single array
    # to keep track of which indices should be included in our dataset.
    final_map = None

    for label, location in pairs:
        # Determine which indices satisfy both of our criteria, and then create
        # a map for both using the AND operator.
        location_map = train_dataset.metadata_array[:,0] == location
        label_map = train_dataset.y_array == label

        both_map = np.logical_and(location_map, label_map)

        if final_map is None:
            final_map = both_map
        else:
            final_map = np.logical_or(final_map, both_map)

    final_idx = np.where(final_map)[0]

    # Create a WILDSSubset object using the train_dataset and these indices.
    # Include a transform that converts the images to tensors, resizes them,
    # and then normalize them according to the IMAGENET normalization values.
    # More details can be found here: https://pytorch.org/vision/stable/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    composed = transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor(), normalize])
    return WILDSSubset(train_dataset, final_idx, composed)
