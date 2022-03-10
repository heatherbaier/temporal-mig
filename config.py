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


training_args = add_argument_group("Training Arguments")
# training_args.add_argument("--batch_size", 
#                            type = int, 
#                            default = 20, 
#                            help = "batch size is equivalent to number of process to run asychronously / nodes * ppn")
training_args.add_argument("--tv_split", 
                           type = float, 
                           default = .75, 
                           help = "Train/Val split percentage - given as a float i.e. .75")
training_args.add_argument("--num_epochs", 
                           type = int, 
                           default = 100, 
                           help = "Number of epochs")
training_args.add_argument("--n_mini_epochs", 
                           type = int, 
                           default = 2, 
                           help = "Number of mini epochs performed on each image within an epoch")
training_args.add_argument("--eps_decay", 
                           type = int, 
                           default = 150, 
                           help = "Value by which to decay epsilon")
training_args.add_argument("--memory_limit", 
                           type = int, 
                           default = 1000, 
                           help = "Value by which to decay epsilon")
training_args.add_argument("--mem_batch_size", 
                           type = int, 
                           default = 4, 
                           help = "Value by which to decay epsilon")
training_args.add_argument("--lr", 
                           type = float, 
                           default = .01, 
                           help = "Learning Rate")
training_args.add_argument("---ppn", 
                           type = float, 
                           default = 32, 
                           help = "Learning Rate")
training_args.add_argument("--ppn_usable", 
                           type = float, 
                           default = 32, 
                           help = "Learning Rate")


env_args = add_argument_group("RL Environment Arguments")
env_args.add_argument("--display", 
                      type = str, 
                      default = "False", 
                      help = "Whether to display the interactive RL environment during training - Cannot display on HPC")


model_args = add_argument_group("Model Arguments")
model_args.add_argument("--n_glimpses", 
                      type = str, 
                      default = 4, 
                      help = "Size of glimpse_net FC layer output")
model_args.add_argument("--glimpse_hidden_size", 
                      type = str, 
                      default = 128, 
                      help = "Size of glimpse_net FC layer output")
model_args.add_argument("--core_hidden_size", 
                      type = str, 
                      default = 128, 
                      help = "Size of glimpse_net FC layer output")
model_args.add_argument("--lc_fc_size", 
                      type = int, 
                      default = 128,
                      help = "How many land cover glimpses the RL agent can view")


data_args = add_argument_group("Data Arguments")
data_args.add_argument("--imagery_dir",
                      type = str,
                      default = "/sciclone/scr-mlt/hmbaier/cropped/",
                      help = "Full path to directory containing imagery")
# data_args.add_argument("--lc_dir",
#                       type = str,
#                       default = "/sciclone/geograd/Heather/lc/extracted_100m/",
#                       help = "Full path to land cover tiff file directory")
# data_args.add_argument("--y_path",
#                       type = str,
#                       default = "/sciclone/home20/hmbaier/claw/migration_data.json",
#                       help = "Full path to json containing muni_id -> num_migrants mapping.")
# data_args.add_argument("--census_path",
#                       type = str,
#                       default = "/sciclone/geograd/Heather/lc/census_feats,json",
#                       help = "Full path to json containing muni_id -> census features mapping.")
# data_args.add_argument("--lc_path",
#                       type = str,
#                       default = "/sciclone/geograd/Heather/lc/lc_feats.json",
#                       help = "Full path to json containing muni_id -> land cover features mapping.")
# data_args.add_argument("--lc_map",
#                       type = str,
#                       default = "/sciclone/geograd/Heather/lc/lc_map.json",
#                       help = "Full path to json containing lc_id -> land cover type mapping.")

now = datetime.now()

misc_args = add_argument_group("Miscellaneous Arguments")
misc_args.add_argument("--log_name",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ")/log (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ").txt",
                      help = "Full path to directory containing imagery")
misc_args.add_argument("--train_records_name",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ")/train_records_epoch{epoch} (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ").txt",
                      help = "Full path to directory containing imagery")
misc_args.add_argument("--val_records_name",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ")/val_records_epoch{epoch} (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ").txt",
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
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ")/",
                      help = "Path to save epoch records too")
misc_args.add_argument("--epochs_dir",
                      type = str,
                      default = "/sciclone/home20/hmbaier/tm/records/records (" + str(date.today()) + ", " + str(datetime.strptime(now.strftime("%H:%M"), "%H:%M").strftime("%I:%M %p")) + ")/epochs/",
                      help = "Path to save epoch records too")



# glimpse network params
glimpse_arg = add_argument_group("Glimpse Network Params")
glimpse_arg.add_argument(
    "--patch_size", type=int, default = 50, help="size of extracted patch at highest res"
)
glimpse_arg.add_argument(
    "--glimpse_scale", type=int, default = .75, help="scale of successive patches"
)
glimpse_arg.add_argument(
    "--num_patches", type=int, default = 2, help="# of interpolated patches per glimpse"
)
glimpse_arg.add_argument(
    "--loc_hidden", type=int, default = 128, help="hidden size of loc fc"
)
glimpse_arg.add_argument(
    "--glimpse_hidden", type=int, default = 128, help="hidden size of glimpse fc"
)


# core network params
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--num_glimpses", type=int, default = 4, help="# of glimpses, i.e. BPTT iterations"
)
core_arg.add_argument("--hidden_size", type = int, default = 256, help="hidden size of rnn")


# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument(
    "--std", type=float, default = 0.1, help="gaussian policy standard deviation"
)
reinforce_arg.add_argument(
    "--M", type=int, default=1, help="Monte Carlo sampling for valid and test sets"
)


# data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=128, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=True,
    help="Whether to shuffle the train and valid indices",
)
data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=False,
    help="Whether to visualize a sample grid of the data",
)


# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)
train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default = 10, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)
train_arg.add_argument(
    "--train_patience",
    type=int,
    default=50,
    help="Number of epochs to wait before stopping train",
)


# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type = str2bool, default = False, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./data", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    type=str2bool,
    default=False,
    help="Whether to use tensorboard for visualization",
)
misc_arg.add_argument(
    "--resume",
    type=str2bool,
    default=False,
    help="Whether to resume training from checkpoint",
)
misc_arg.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="How frequently to print training details",
)
misc_arg.add_argument(
    "--plot_freq", type=int, default=1, help="How frequently to plot glimpses"
)



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed




