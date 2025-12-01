import yaml

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu
    "devices": ["cuda:0"],
    "dtype": "fp32",
    # DDPM or FM
    "model_type": "DDPM",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    "dataset": {
        "name": "MNIST",
        # training / testing splits
        "split": "train",
        "data_folder": None,
    },
    "loader": {
        # batch size
        "batch_size": 64,
        # number of workers for data fetching
        "num_workers": 4,
    },

    # unet architecture
    "model": {
        # shape of input image (C x H W)
        "img_shape": None,
        # number of timesteps in the diffusion process
        "timesteps": 1000,
        # number of classes used for conditioning
        "num_classes": None,
        # base feature dimension in UNet
        "dim": 64,
        # condition dimension (embedding of the label) in UNet
        "context_dim": 64,
        # multiplier of feature dimensions in UNet
        # length of this list also specifies #blockes in UNet encoder/decoder
        # e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x
        # w.r.t. base feature dim
        "dim_mults": (1, 2, 4),
        # attn_levels (tuple/list of int): specify if attention layer is included
        # in a block in the UNet encoder/decoder
        # e.g., (0, 1) -> the first two blocks in the encoder and the last two
        "attn_levels": (0, 1),
        # if to use VAE for latent diffusion model
        "use_vae": False,
        "vae_encoder_weights": None,
        "vae_decoder_weights": None,
    },
    # training config
    "train_cfg": {
        # learning rate
        "lr": 2e-5,
        # number of epochs in training
        "epochs": 30,
        # #samples (per category) to visualize when evaluating the model
        "num_eval_viz_samples": 10,

    },
    # test config
    "test_cfg": {
        # test data folder
        "test_data_folder": None,
        # #samples (per category) to draw for evaluating the model
        "num_eval_samples": 100,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    return config
