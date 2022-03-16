from collections import OrderedDict
from torchvision import models
import torch
import gc


def load_extracter_state(state_dict):

    r18 = models.resnet18()
    r18.fc = torch.nn.Linear(512, 1)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if "fc." not in key:
            new_key = key_transformation[key]
            new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict