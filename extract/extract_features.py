from collections import OrderedDict
from torchvision import models
import argparse
import torch
import json
import gc

from extractloader import *
from resnet18 import *
from utils import *


if __name__  == "__main__":
    
    ######################################################
    # MPI-esque
    ######################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", help="Country ISO")
    parser.add_argument("world_size", help="ADM level")
    args = parser.parse_args()

    ######################################################
    # Load in model and saved weights
    ######################################################   
    model = r18(models.resnet18())
    model_path = "/sciclone/home20/hmbaier/tm/records/records (2022-03-16, 13:22:22)/models/model_epoch1.torch"
    weights = torch.load(model_path)["model_state_dict"]
    sd = load_extracter_state(weights)
    model.load_state_dict(sd)

    ######################################################
    # Make list of dates for JSON
    ######################################################  
    dates = []
    for year in range(1995, 2001):
        for month in range(1, 13):
            dates.append(str(year) + "-" + str(month))
    
    ######################################################
    # Make list of munis and grab rank specific subset
    ######################################################  
    imagery_dir = "/sciclone/scr-mlt/hmbaier/cropped/"
    imagery_list = os.listdir(imagery_dir)
    imagery_list = [imagery_dir + i for i in imagery_list if "484" in i]
    imagery_list = np.array_split(imagery_list, int(args.world_size))
    rank_munis = imagery_list[int(args.rank) - 1]

    output_dir = "/sciclone/geograd/heather_data/temporal_features/jsons/"

    for imname in rank_munis:

        print(imname)

        data = ExtractLoader(muni = imname)

        features = {}
        for c, (im, migs) in enumerate(zip(data.imagery, data.migs)):
            im = im.permute(0,3,1,2)
            output = model(im)[0]
            features[dates[c]] = {"migrants": str(migs.item()), "features: ": str(list(output.detach().numpy()))}

        with open(output_dir + data.muni_name + ".json", "w") as f:
            json.dump(features, f)
