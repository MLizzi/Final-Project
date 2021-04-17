import dataloading
import models

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

if __name__ == '__main__':
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')
    # partition_set = dataloading.collect_groups(train_data, [11,12,13], [idx for idx in range(100)])
    partition_set = dataloading.collect_location_label_pairs(train_data, [(idx, idx) for idx in range(5)])
    partition_loader = get_train_loader('standard', partition_set, batch_size=16)

    model = models.load_modified_pre_trained('resnet18', 5)

    models.fine_tune_model(model, partition_loader, "./test_model", num_epochs=10)

    print(partition_set.metadata_array.shape)
    print(train_data.metadata_array.shape)

    # Set up the testing data.
    #train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    #train_loader = get_train_loader('standard', train_data, batch_size=16)

    #grouper = CombinatorialGrouper(dataset, ['location'])

"""
# To allow algorithms to leverage domain annotations as well as other groupings over the available metadata, the WILDS
# package provides Grouper objects. These Grouper objects extract group annotations from metadata, allowing users to
# specify the grouping scheme in a flexible fashion.

>>> from wilds.common.grouper import CombinatorialGrouper

# Initialize grouper, which extracts domain information
# In this example, we form domains based on location
>>> grouper = CombinatorialGrouper(dataset, ['location'])

# Train loop
>>> for x, y_true, metadata in train_loader:
...   z = grouper.metadata_to_group(metadata)
...   ...

# The Grouper can be used to prepare a group-aware data loader that, for each minibatch, first samples a specified
# number of groups, then samples examples from those groups. This allows our data loaders to accommodate a wide array
# of training algorithms, some of which require specific data loading schemes.

# Prepare a group data loader that samples from user-specified groups
>>> train_loader = get_train_loader('group', train_data,
...                                 grouper=grouper,
...                                 n_groups_per_batch=2,
...                                 batch_size=16)

# Example:
# https://github.com/p-lambda/wilds/blob/b38304bb6ac3b3f9326cf028d77be9f0cb7c8cdb/examples/algorithms/initializer.py
"""
