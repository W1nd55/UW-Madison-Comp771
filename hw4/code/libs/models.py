from .ddpm import DDPM
from .fm import FM

def build_model(model_type, model_cfg):
    """a help function to create DDPM / FM models"""
    if "DDPM" in model_type:
        model = DDPM(**model_cfg)
        return model
    elif "FM" in model_type:
        model = FM(**model_cfg)
        return model
    else:
        raise ValueError("Unsupported model type!")
