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
    composed = transforms.Compose([normalize, transforms.Resize((448,448)), transforms.ToTensor()])

    return WILDSSubset(train_dataset, both_idx, composed)
