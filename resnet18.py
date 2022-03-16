from collections import OrderedDict
from torchvision import models
import torch
import gc

class r18(torch.nn.Module):

    def __init__(self, resnet):
        super().__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.flatten(x, start_dim = 1)
        return x


def load_ddp_state(state_dict):

    r18 = models.resnet18()
    r18.fc = torch.nn.Linear(512, 1)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation[key]
        new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict


if __name__ == "__main__":

    model = r18(models.resnet18())

    fname = "/sciclone/home20/hmbaier/tm/records/records (2022-03-15, 15:15:31)/models/model_epoch0.torch"
    
    weights = torch.load(fname)["model_state_dict"]

    sd = load_ddp_state(weights)

    print(sd)
