import argparse
import random
import numpy as np

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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_type",
                        help="The type of model.")
    parser.add_argument("output_model_path",
                        help="Where to save the model.")
    # Optional arguments
    parser.add_argument("-b",
                        default=80,
                        help="Batch size.")
    # Optional arguments
    parser.add_argument("-e",
                        default=10,
                        help="Num epochs.")
    args = parser.parse_args()

    # Some torch set up
    seed = 1234#100#1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Default device + GPU random seed setting
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        device = torch.device("cpu")

    dataset = get_dataset(dataset='iwildcam', download=True)
    train_data = dataset.get_subset('train')

    # Partition the locations so that no two labels share a location.
    analyze_dataset.create_location_array(train_data, "./data/location_counts.csv")
    location_lists = analyze_dataset.greedy_location_split("./data/location_counts.csv", 2000)

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

    partition_loader = get_train_loader('standard', partition_set, batch_size=int(args.b), worker_init_fn=seed_worker)
    eval_loader = get_eval_loader('standard', partition_test_set, batch_size=16)
        
    # partition_set = dataloading.collect_location_label_pairs(train_data, [(idx, idx) for idx in range(5)])
    # partition_loader = get_train_loader('standard', partition_set, batch_size=16)

    model = models.load_modified_pre_trained(args.input_model_type, 4)

    # Place model on device
    model = model.to(device)

    models.fine_tune_model(model, partition_loader, eval_loader, args.output_model_path, device, num_epochs=int(args.e))

    # model.load_state_dict(torch.load("./seed_test"))

    print(partition_set.metadata_array.shape)
    print(train_data.metadata_array.shape)

    # Set up the testing data.
    #train_data = dataset.get_subset('train', transform=transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor()]))
    #train_loader = get_train_loader('standard', train_data, batch_size=16)

    #grouper = CombinatorialGrouper(dataset, ['location'])
