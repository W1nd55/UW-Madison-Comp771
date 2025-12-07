from .ddpm import DDPM
from .fm import FM
from .ddpm_edm import EDM, EDMStochastic

def build_model(model_type, model_cfg):
    """a help function to create DDPM / FM / EDM models"""
    if model_type == "EDM":
        model = EDM(**model_cfg)
        return model
    elif model_type == "EDMStochastic":
        model = EDMStochastic(**model_cfg)
        return model
    elif "DDPM" in model_type:
        model = DDPM(**model_cfg)
        return model
    elif "FM" in model_type:
        model = FM(**model_cfg)
        return model
    else:
        raise ValueError("Unsupported model type!")
