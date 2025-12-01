# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

# for visualization
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# our code
from libs import (
    load_config,
    build_dataset,
    build_dataloader,
    build_model,
    save_checkpoint,
    ModelEMA,
    AverageMeter
)


################################################################################
def main(args):
    """main function that handles training"""

    """1. Load config / Setup folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(cfg["output_folder"], cfg_filename + "_" + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg["output_folder"], cfg_filename + "_" + str(args.output)
        )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    """2. create dataset / dataloader"""
    # dataset and update model parameters
    train_dataset, num_classes, img_shape = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["split"],
        cfg["dataset"]["data_folder"],
    )
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["img_shape"] = img_shape
    pprint(cfg)
    # data loaders
    train_loader = build_dataloader(train_dataset, **cfg["loader"])

    """3. create model and optimizer"""
    # model
    model = build_model(
        cfg['model_type'], cfg['model']
    ).to(torch.device(cfg["devices"][0]))
    # set model to training
    model.train()
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train_cfg"]["lr"])
    # enable cudnn benchmark
    cudnn.benchmark = True

    # enable model EMA (appendix B of DDPM paper)
    model_ema = ModelEMA(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(
                args.resume,
                map_location=cfg["devices"][0],
                weights_only=True
            )
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{:s}' (epoch {:d})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training loop"""
    print("\nStart training ...")

    # start training
    max_epochs = cfg["train_cfg"]["epochs"]
    device = torch.device(cfg["devices"][0])

    start = time.time()
    batch_time = AverageMeter()
    loss_tracker = AverageMeter()
    num_iters_per_epoch = len(train_loader)

    # start training
    if cfg["dtype"] == "fp16":
        scaler = GradScaler()
        print("Using mixed precision training")
    else:
        scaler=None

    # As DDPM training samples the time step for each sample, epoch only controls
    # the total number of iterations (and not the actual epochs)
    for epoch in range(args.start_epoch, max_epochs):
        # reset the iters (so that resuming is possible)
        total_iters = num_iters_per_epoch * epoch

        # the main training loop
        for iter_idx, batch in enumerate(train_loader):
            # fetching imgs and their labels
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            # zero out the optmizer
            optimizer.zero_grad()

            # if mixed precision training
            if scaler != None:
                # compute loss
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.compute_loss(img, label)
                # update the optimizer / scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # compute loss
                loss = model.compute_loss(img, label)

                # backward loss and update the parameters
                loss.backward()
                optimizer.step()

            # update the EMA version of the model
            model_ema.update(model)
            total_iters += 1

            # print the loss values
            if (total_iters % args.print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / args.print_freq)
                start = time.time()
                # record the loss value
                loss_tracker.update(loss.item())
                # log to tensorboard
                tb_writer.add_scalar(
                    "training/loss", loss_tracker.val, total_iters
                )
                # print to terminal
                block1 = "Iter {:08d}".format(total_iters)
                block2 = "Time {:.2f} ({:.2f})".format(
                    batch_time.val, batch_time.avg
                )
                block3 = "Loss {:.3f} ({:.3f})".format(
                    loss_tracker.val, loss_tracker.avg
                )
                print("\t".join([block1, block2, block3]))

        # save ckpt / draw samples once in a while
        if ((epoch + 1) == max_epochs) or (
            (args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)
        ):
            save_states = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            is_final = (epoch + 1) == max_epochs
            save_checkpoint(
                save_states,
                is_final,
                file_folder=ckpt_folder,
                file_name="epoch_{:03d}.pth.tar".format(epoch),
            )

            # draw samples from the current model at the end of every epoch
            sample_labels = torch.arange(0, num_classes, dtype=torch.long).to(device)
            all_imgs = []

            for _ in range(cfg["train_cfg"]["num_eval_viz_samples"]):
                # Samples are drawn from EMA version of the model.
                # This will initially produce samples lagged behind current epoch,
                # yet eventually stablize samples during late stage of training
                imgs = model_ema.module.generate(sample_labels)
                all_imgs.append(imgs)

            all_imgs = torch.cat(all_imgs, dim=0)
            save_image(
                all_imgs,
                os.path.join(ckpt_folder, 'sample-{:d}.png'.format(epoch)),
                nrow=cfg["model"]["num_classes"]
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train DDPM/FM for Image Generation"
    )
    parser.add_argument("config", metavar="DIR", help="path to a config file")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=100,
        type=int,
        help="print frequency (default: 100 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=10,
        type=int,
        help="checkpoint frequency (default: every 5 epochs)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)
