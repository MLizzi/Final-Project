"""Code for setting up, modifying, and training (?) models."""

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


def fine_tune_model(model, train_dataloader, exper_dir, num_epochs=20):
    """Fine tune a pretrained model.

    Parameters:
    model: Torch Model. The model to be trained, should output
        a *num_classes* vector for each input.
    train_dataloader: torch dataloader. Should iterate over batches
        of (training_data, training_labels).
    exper_name: str. Path to location where model should be saved.
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
        for x_train, y_train in train_dataloader:
            optim.zero_grad()

            y_hat = model(x_train)
            
            loss = loss_func(y_hat, y_train)

            # Store the losses so that we can track training progress.
            loss_list.append(loss.item())

            loss.backward()

            optim.step()

        print("Epoch {}. Training Loss: {}".format(epoch, np.mean(loss_list)))

    # Save the trained model to the specified directory.
    torch.save(model.state_dict(), exper_dir)

    return model
