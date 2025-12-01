# python imports
import argparse
import os
import time
import glob
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# for visualization
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

# our code
from libs import (
    build_dataset,
    load_config,
    build_model,
    compute_fid_score
)


################################################################################
def main(args):
    """main function that handles training"""

    """1. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, "epoch_{:03d}.pth.tar".format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, "*.pth.tar")))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    """2. create model"""
    # dataset and update model parameters
    _, num_classes, img_shape = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["split"],
        cfg["dataset"]["data_folder"],
    )
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["img_shape"] = img_shape
    pprint(cfg)

    # model
    model = build_model(
        cfg['model_type'], cfg['model']
    ).to(torch.device(cfg["devices"][0]))
    # set model to evaluation
    model.eval()
    # enable cudnn benchmark
    cudnn.benchmark = True

    """3. load ckpt and setup output folder"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt
    checkpoint = torch.load(
        ckpt_file,
        map_location=cfg["devices"][0],
        weights_only=True
    )
    print("Loading model ...")
    # load the EMA version of the model
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint

    # setup output folder and subfolders
    output_folder = os.path.join(
        os.path.dirname(ckpt_file),
        os.path.basename(ckpt_file).rstrip(".pth.tar")
    )
    os.makedirs(output_folder, exist_ok=True)
    for cat_idx in range(num_classes):
        os.makedirs(
            os.path.join(output_folder, 'cat-{:d}'.format(cat_idx)),
            exist_ok=True
        )

    """4. Generating Images for Evaluation"""
    print("\nGenerating images ...")

    # draw samples from the model
    device = torch.device(cfg["devices"][0])
    sample_labels = torch.arange(0, num_classes, dtype=torch.long).to(device)

    for sample_idx in tqdm(range(cfg["test_cfg"]["num_eval_samples"])):
        # Samples are drawn from EMA version of the model.
        imgs = model.generate(sample_labels)
        for cat_idx, img in enumerate(imgs):
            output_file = os.path.join(
                output_folder,
                'cat-{:d}'.format(cat_idx),
                '{:d}.png'.format(sample_idx)
            )
            save_image(img, output_file)

    # comptue FID scores
    print("\nEvaluating FID scores ...")
    fid_socre = compute_fid_score(
        cfg["test_cfg"]["test_data_folder"], output_folder
    )
    print("FID: ", fid_socre)
    return

################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Evaluate DDPM/FM for Image Generation"
    )
    parser.add_argument("config", type=str, metavar="DIR", help="path to a config file")
    parser.add_argument("ckpt", type=str, metavar="DIR", help="path to a checkpoint")
    parser.add_argument("-e", "--epoch", type=int, default=-1, help="checkpoint epoch")
    args = parser.parse_args()
    main(args)
