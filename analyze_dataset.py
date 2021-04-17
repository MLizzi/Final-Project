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


def greedy_location_split(file_path, num_samples):
    """Partition locations for each of the 4 classes.

    Note: Since this method uses a greedy allocation
        strategy, it is possible that *num_samples* will not
        be found for each class. In that case, a ValueError is
        raised.

    Parameters:
    file_path: str. The path to a .csv file created by `create_location_array`.
    num_samples: int. The min number of samples that each class should have.

    returns: 4-tuple of lists of ints. The lists of locations
        for each of the classes.

    """
    dataframe = pd.read_csv(file_path)

    # Keep track of which locations have been claimed.
    taken_locations = []
    location_lists = []
    for label in ['Boar', 'Rodent', 'Puma', 'Turkey']:
        # Sort the dataframe using this index.
        label_array = dataframe.sort_values(by=[label], ascending=False)
        # print(label_array)
        # Iterate until we've collected *num_samples*, or until
        # we've been over every row, whichever comes first.
        samples_stored = 0
        label_locations = []
        for index, row in label_array.iterrows():
            #print(label_locations)
            if row['Location'] not in taken_locations:
                taken_locations.append(row['Location'])
                label_locations.append(row['Location'])
                samples_stored += row[label]

            # Check if we have enough samples
            if samples_stored >= num_samples:
                break
        
        # Raise an error if we didn't get enough samples.
        if samples_stored < num_samples:
            raise ValueError('{} did not have enough samples!'.format(label))

        location_lists.append(label_locations)

        print("{}: {} samples".format(label, samples_stored))

    print(location_lists)
    return location_lists 


if __name__ == "__main__":
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')
    #create_location_array(train_data, "./location_counts.csv")
    greedy_location_split("./location_counts.csv", 2000)
