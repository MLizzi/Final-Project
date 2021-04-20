import argparse

import dataloading
import models


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import torch
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model_path",
                        help="The path to the input CSV.")
    parser.add_argument("input_model_type",
                        help="The type of model.")
    # Optional arguments
    parser.add_argument("-b",
                        default=80,
                        help="Batch size.")

    args = parser.parse_args()
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

    # Set up test set
    test_pair_list = [(1, 'all'), (2, 'all'), (4, 'all'), (146, 'all')]
    partition_test_set = dataloading.collect_location_label_pairs(dataset, test_pair_list, 'test', 'val')
    eval_dataloader = get_eval_loader('standard', partition_test_set, batch_size=int(args.b))

    # Load model.
    print(args.input_model_path)
    model = models.load_modified_pre_trained(args.input_model_type, 4)

    # Place model on device
    model = model.to(device)

    # Load the weights
    model.load_state_dict(torch.load(args.input_model_path))

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = []
        true_labels = []
        for x_test, y_test, test_metadata in eval_dataloader:
            # Place tensors on the correct device and fix labels.
            x_test = x_test.to(device)
            # Don't need to send y_test to the gpu because we don't need
            # to involve the model!
            # y_test = y_test.to(device)
            y_test[y_test == 1] = 0
            y_test[y_test == 2] = 1
            y_test[y_test == 4] = 2
            y_test[y_test == 146] = 3

            preds = model(x_test)
            predictions.append(preds.cpu())
            true_labels.append(y_test)

    all_preds = torch.argmax(torch.cat(predictions), dim=1)
    all_true_labels = torch.cat(true_labels)

    test_accuracy = accuracy_score(all_true_labels, all_preds)

    print("Test Accuracy: {}".format(test_accuracy))
    f1 = f1_score(all_true_labels, all_preds, average=None)
    print(f1)

    print("Boar {}".format(partition_test_set.metadata_array[partition_test_set.metadata_array[:,-1] == 1].shape))
    print("Rodent {}".format(partition_test_set.metadata_array[partition_test_set.metadata_array[:,-1] == 2].shape))
    print("Puma {}".format(partition_test_set.metadata_array[partition_test_set.metadata_array[:,-1] == 4].shape))
    print("Turkey {}".format(partition_test_set.metadata_array[partition_test_set.metadata_array[:,-1] == 146].shape))
