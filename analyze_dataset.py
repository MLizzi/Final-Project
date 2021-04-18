""" Methods for analyzing the WILDS dataset."""
import pandas as pd
import torch
import numpy as np

from wilds import get_dataset


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
        count_dict['Std'] = np.std([count_dict['Boar'], count_dict['Rodent'],
                                    count_dict['Puma'], count_dict['Turkey']])

        boar_here = 1 if count_dict['Boar'] != 0 else 0
        rodent_here = 1 if count_dict['Rodent'] != 0 else 0
        puma_here = 1 if count_dict['Puma'] != 0 else 0
        turkey_here = 1 if count_dict['Turkey'] != 0 else 0

        overlap = boar_here + rodent_here + puma_here + turkey_here
        count_dict['Overlap'] = overlap
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


def find_overlap_locations(file_path, num_samples):
    """Find a locations that have a high concentration of all 4 labels.

    Parameters:
    file_path: str. The path to a .csv file created by `create_location_array`.
    num_samples: int. The min number of samples that each class should have.

    returns: 4-tuple of lists of ints. The lists of locations
        for each of the classes.

    """
    # For now, just select 
    dataframe = pd.read_csv(file_path)

    # Keep track of which locations have been claimed.
    boar_list = []
    boar_num = 0
    rodent_list = []
    rodent_num = 0
    puma_list = []
    puma_num = 0
    turkey_list = []
    turkey_num = 0

    sorted_by_std = dataframe.sort_values(by=['Std'])

    for index, row in sorted_by_std.iterrows():
        if row['Overlap'] > 1:

            if boar_num < num_samples and row['Boar'] != 0:
                boar_list.append(row['Location'])
                boar_num += row['Boar']
            if rodent_num < num_samples and row['Rodent'] != 0:
                rodent_list.append(row['Location'])
                rodent_num += row['Rodent']
            if puma_num < num_samples and row['Puma'] != 0:
                puma_list.append(row['Location'])
                puma_num += row['Puma']
            if turkey_num < num_samples and row['Turkey'] != 0:
                turkey_list.append(row['Location'])
                turkey_num += row['Turkey']
        print("{} {} {} {}".format(boar_num, rodent_num, puma_num, turkey_num))

        if boar_num >= num_samples and rodent_num >= num_samples and \
                puma_num >= num_samples and turkey_num >= num_samples:
            break

    if boar_num < num_samples or rodent_num < num_samples or \
            puma_num < num_samples or turkey_num < num_samples:
        raise ValueError('Could not satisfy num_samples!')

    return [boar_list, rodent_list, puma_list, turkey_list]


if __name__ == "__main__":
    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')
    # create_location_array(train_data, "./location_counts.csv")
    greedy_location_split("./location_counts.csv", 2000)
