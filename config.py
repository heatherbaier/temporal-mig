from datetime import datetime, date
import argparse


arg_lists = []
parser = argparse.ArgumentParser(description = "a3cRAM")

def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


with open("/sciclone/home20/hmbaier/tm/date.txt", "r") as f:
    now = f.read().strip("\n")


training_args = add_argument_group("Training Arguments")
training_args.add_argument("--tv_split", 
                           type = float, 
                           default = 0.6, 
                           help = "Train/Val split percentage - given as a float i.e. .75")
training_args.add_argument("--lr", 
                           type = float, 
                           default = .0001, 
                           help = "Learning Rate")


data_args = add_argument_group("Data Arguments")
data_args.add_argument("--imagery_dir",
                      type = str,
                      default = "/sciclone/scr-mlt/hmbaier/cropped/",
                      help = "Full path to directory containing imagery")


misc_args = add_argument_group("Miscellaneous Arguments")
misc_args.add_argument("--log_name",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + now + ")/log (" + now + ").txt",
                      help = "Full path to directory containing imagery")
misc_args.add_argument("--use_rpc",
                      type = bool,
                      default = False,
                      help = "Whether to use RPC to communicate training statistics.")
misc_args.add_argument("--find_unused_parameters",
                      type = bool,
                      default = False,
                      help = "Whether to use find unused parameters in DDP model initialization.")
misc_args.add_argument("--records_dir",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + now + ")/",
                      help = "Path to save epoch records too")
misc_args.add_argument("--epochs_dir",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + now + ")/epochs/",
                      help = "Path to save epoch records too")
misc_args.add_argument("--models_dir",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + now + ")/models/",
                      help = "Path to save trained_models too")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed