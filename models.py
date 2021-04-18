"""Code for setting up, modifying, and training (?) models."""

from sklearn.metrics import accuracy_score
import torch
import torchvision
import numpy as np

def load_modified_pre_trained(model_name, num_output):
    """Load in a specified pre trained model and add a new final layer.

    Parameters:
    model_name: str. The name of the requested model.
    num_output: int. The new number of outputs for this model.

    """
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        # Swap out the final layer.
        new_linear = torch.nn.Linear(model.fc.in_features, num_output)
        model.fc = new_linear
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        # Swap out the final layer.
        new_linear = torch.nn.Linear(model.fc.in_features, num_output)
        model.fc = new_linear
    else:
        raise ValueError("{} is not a valid model selection.".format(model_name))

    return model


def fine_tune_model(model, train_dataloader, eval_dataloader, exper_dir, device, num_epochs=20):
    """Fine tune a pretrained model.

    Parameters:
    model: Torch Model. The model to be trained, should output
        a *num_classes* vector for each input.
    train_dataloader: torch dataloader. Should iterate over batches
        of (training_data, training_labels, metadata).
    eval_loader: torch dataloader. Should iterate over batches
        of (training_data, training_labels, metadata).
    exper_name: str. Path to location where model should be saved.
    device: torch device. The device to place tensors on.
    num_epochs: int. The number of epochs to train for.
    """
    # Training criteria and optimizer.
    loss_func = torch.nn.CrossEntropyLoss()
    # Weight decay and momentum following the 
    # "Deep Residual Learning for Image Recognition" paper.
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)

    # TODO: Add LR scheduler?

    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        accuracy_list = []
        for x_train, y_train, metadata in train_dataloader:
            optim.zero_grad()

            # Place tensors on the correct device and fix labels.
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_train[y_train == 1] = 0
            y_train[y_train == 2] = 1
            y_train[y_train == 4] = 2
            y_train[y_train == 146] = 3

            y_hat = model(x_train)
            
            loss = loss_func(y_hat, y_train)

            # Store the losses so that we can track training progress.
            loss_list.append(loss.item())

            loss.backward()

            optim.step()

            # Calculate the accuracy as well.
            train_preds = torch.argmax(y_hat, dim=1)
            accuracy_list.append(accuracy_score(y_train.cpu(), train_preds.cpu()))
            print(loss.item())

        """ Commented out because of memory issues.
        # Before doing the evaluation, delete all training related variables off
        # of the GPU to save memory.
        del y_hat
        del x_train
        del y_train
        del train_preds
        torch.cuda.empty_cache()
    
        # Evaluate the model
        model.eval()
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

        print("Epoch {}. Training Loss: {}, Training Accuracy: {}".format(
            epoch, np.mean(loss_list), np.mean(accuracy_list)))
        print("Test Accuracy: {}".format(test_accuracy))
        """

    # Save the trained model to the specified directory.
    torch.save(model.state_dict(), exper_dir)

    return model
