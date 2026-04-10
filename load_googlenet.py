def load_googlenet(weight_path, num_classes, device):
    import torch
    import torch.nn as nn
    from torchvision import models

    # Disable auxiliary logits for inference
    model = models.googlenet(pretrained=False, aux_logits=False)

    # Replace final classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load trained weights
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model
