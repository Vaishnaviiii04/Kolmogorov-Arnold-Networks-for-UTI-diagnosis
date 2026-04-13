import torch
from models.pretrained_kan_c_mlp import KANCMLP

def load_kan_model(weight_path, device="cpu"):
    model = KANCMLP(num_classes=3, device=device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model
