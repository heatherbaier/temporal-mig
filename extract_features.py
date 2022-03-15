import torch




def load_model(fname):

    state = torch.load(fname)
    return state["model_state_dict"]

def main()