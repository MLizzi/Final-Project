"""Code for setting up, modifying, and training (?) models."""

import torch
import torchvision
import numpy

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
