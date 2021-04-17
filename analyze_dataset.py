""" Methods for analyzing the WILDS dataset."""
import pandas as pd
import torch

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

def create_location_array(train_data, file_path):
    """ Create a csv file detailing the amount of samples at each location.

    Parameters:
    train_data: WILDSDataset object. The training data.
    file_path: str. The path to where the csv should be saved. Should end
        with '.csv'.

    """
    # List of labels that we're interested in.
    # Boar, rodent, puma, and turkey.
    label_list = [1, 2, 4, 146]

    # Determine the number of locations
    location_num = torch.amax(train_data.metadata_array[:,0])

    # For each location, determine how many samples with one of the
    # target labels occur at that location.
    dict_list = []
    for loc_idx in range(location_num + 1):
        # Extract all the data at this location
        loc_data = train_data.metadata_array[train_data.metadata_array[:,0] == loc_idx]
        count_dict = {
            'Location': loc_idx,
            'Boar': loc_data[loc_data[:,-1] == label_list[0]].shape[0],
            'Rodent': loc_data[loc_data[:,-1] == label_list[1]].shape[0],
            'Puma': loc_data[loc_data[:,-1] == label_list[2]].shape[0],
            'Turkey': loc_data[loc_data[:,-1] == label_list[3]].shape[0]
        }
        dict_list.append(count_dict)

    # Convert list of dictionaries into CSV.
    pd.DataFrame(dict_list).to_csv(file_path)


if __name__ == "__main__":
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')
    create_location_array(train_data, "./location_counts.csv")
