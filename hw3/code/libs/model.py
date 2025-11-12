import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, xs):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        outs = []
        for x in xs:
            y = self.conv(x)
            outs.append(self.cls_logits(y))
        return outs


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)
        nn.init.constant_(self.bbox_ctrness.bias, 0.0)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, xs):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        reg_outs, ctr_outs = [], []
        for x in xs:
            y = self.conv(x)
            reg_outs.append(self.bbox_reg(y))
            ctr_outs.append(self.bbox_ctrness(y))
        return reg_outs, ctr_outs


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)
        
        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        """
        Compute FCOS losses including classification, regression, and centerness.
        """
        # points: (5,H,W,2) list
        # strides: (5,) tensor
        # reg_range: (5,2) tensor
        # cls_logits: (5, N,C,H,W) list
        # reg_outputs: (5, N,4,H,W) list
        # ctr_logits: (5, N,1,H,W) list
        device = cls_logits[0].device
        num_levels = len(cls_logits)
        B = cls_logits[0].shape[0]
        C = self.num_classes

        # ---- 1) Flatten per-level predictions to [B, sum(L_i), ...], and prep point meta ----
        cls_flat, reg_flat, ctr_flat = [], [], []
        all_points, all_strides, all_lo, all_hi = [], [], [], []

        for i in range(num_levels):
            # logits: [B, C, H, W] -> [B, HW, C]
            cls_i = cls_logits[i].permute(0, 2, 3, 1).reshape(B, -1, C)
            # reg: [B, 4, H, W] -> [B, HW, 4]
            reg_i = reg_outputs[i].permute(0, 2, 3, 1).reshape(B, -1, 4)
            # ctr: [B, 1, H, W] -> [B, HW]
            ctr_i = ctr_logits[i].permute(0, 2, 3, 1).reshape(B, -1)

            cls_flat.append(cls_i)
            reg_flat.append(reg_i)
            ctr_flat.append(ctr_i)

            p = points[i]
            if p.dim() == 3:
                p = p.reshape(-1, 2) # [HW, 2]
            else:
                p = p.view(-1, 2)
            p = p.to(device)
            all_points.append(p)

            Li = p.shape[0]
            stride_i = float(strides[i])  # scalar per level
            all_strides.append(torch.full((Li,), stride_i, device=device, dtype=p.dtype))

            lo_i, hi_i = reg_range[i]
            all_lo.append(torch.full((Li,), float(lo_i), device=device, dtype=p.dtype))
            all_hi.append(torch.full((Li,), float(hi_i), device=device, dtype=p.dtype))

        # cat all levels
        cls_flat = torch.cat(cls_flat, dim=1)            # [B, L, C]
        reg_flat = torch.cat(reg_flat, dim=1)            # [B, L, 4]  (ltrb normalized-by-stride prediction)
        ctr_flat = torch.cat(ctr_flat, dim=1)            # [B, L]
        points_all = torch.cat(all_points, dim=0)        # [L, 2] (x, y in image space)
        strides_all = torch.cat(all_strides, dim=0)      # [L]
        lo_all = torch.cat(all_lo, dim=0)                # [L] in px
        hi_all = torch.cat(all_hi, dim=0)                # [L] in px
        L = points_all.shape[0]

        # ---- 2) Build per-image targets by assignment ----
        cls_t_list, reg_t_list, ctr_t_list, pos_mask_list = [], [], [], []
        total_pos = 0

        for b in range(B):
            t = targets[b]
        
            # boxes in resized image coords (transform already applied): [M, 4] xyxy
            boxes = t["boxes"].to(device)
            # label indexing: expect 0..C-1; if dataset provides 1..C, shift to 0..C-1
            labels = t["labels"].to(device).long()
            M = boxes.shape[0]

            # init targets
            cls_t = torch.zeros((L, C), device=device, dtype=cls_flat.dtype)
            reg_t = torch.zeros((L, 4), device=device, dtype=reg_flat.dtype)  # normalized by stride
            ctr_t = torch.zeros((L,), device=device, dtype=ctr_flat.dtype)
            pos_mask = torch.zeros((L,), device=device, dtype=torch.bool)

            if M == 0:
                cls_t_list.append(cls_t)
                reg_t_list.append(reg_t)
                ctr_t_list.append(ctr_t)
                pos_mask_list.append(pos_mask)
                continue
            
            # points 是 (y, x)
            py = points_all[:, 0].unsqueeze(1)   # y: [L,1]
            px = points_all[:, 1].unsqueeze(1)   # x: [L,1]

            # distances to all boxes
            l = px - boxes[:, 0]                 # [L,M]
            t = py - boxes[:, 1]
            r = boxes[:, 2] - px
            b = boxes[:, 3] - py
            ltrb = torch.stack([l, t, r, b], dim=2)   # [L,M,4]

            # 1) in-box
            in_box = (ltrb.min(dim=2).values > 0)     # [L,M]

            # 2) in-range
            # lo_all / hi_all: [L]，
            max_ltrb = ltrb.max(dim=2).values         # [L,M]
            in_range = (max_ltrb >= lo_all.unsqueeze(1)) & (max_ltrb <= hi_all.unsqueeze(1))  # [L,M]

            # 3) center-sampling
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5    # [M]
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5    # [M]
            rs = self.center_sampling_radius * strides_all  # [L]

            x_mins = torch.maximum(boxes[:, 0].unsqueeze(0), cx.unsqueeze(0) - rs.unsqueeze(1))
            x_maxs = torch.minimum(boxes[:, 2].unsqueeze(0), cx.unsqueeze(0) + rs.unsqueeze(1))
            y_mins = torch.maximum(boxes[:, 1].unsqueeze(0), cy.unsqueeze(0) - rs.unsqueeze(1))
            y_maxs = torch.minimum(boxes[:, 3].unsqueeze(0), cy.unsqueeze(0) + rs.unsqueeze(1))

            in_center_x = (px >= x_mins) & (px <= x_maxs)   # [L,M]
            in_center_y = (py >= y_mins) & (py <= y_maxs)   # [L,M]
            in_center   = in_center_x & in_center_y         # [L,M]

            candidates = in_box & in_range & in_center       # [L, M]

            # choose GT with minimal area among candidates
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [M]
            areas_b = areas.unsqueeze(0).expand(L, M)                            # [L, M]
            INF = torch.tensor(float('inf'), device=device, dtype=areas_b.dtype)
            areas_b = torch.where(candidates, areas_b, INF)

            min_area, min_inds = areas_b.min(dim=1)          # [L]
            pos_mask = torch.isfinite(min_area)              # [L]
            num_pos_b = int(pos_mask.sum().item())
            total_pos += num_pos_b

            if num_pos_b > 0:
                assigned = min_inds[pos_mask]                # [P]
                # classification one-hot
                cls_t[pos_mask] = 0.0
                cls_t[pos_mask, labels[assigned]] = 1.0

                # regression targets (normalize by stride at that point)
                ltrb_pos_px = ltrb[pos_mask, assigned, :]    # [P, 4] in pixels
                reg_t[pos_mask] = (ltrb_pos_px / strides_all[pos_mask].unsqueeze(1)).to(reg_t.dtype)
                # center-ness targets
                l_px, t_px, r_px, b_px = ltrb_pos_px[:, 0], ltrb_pos_px[:, 1], ltrb_pos_px[:, 2], ltrb_pos_px[:, 3]
                
                lr_min = torch.min(l_px, r_px)
                lr_max = torch.max(l_px, r_px)
                tb_min = torch.min(t_px, b_px)
                tb_max = torch.max(t_px, b_px)
                
                # compute centerness scope 0-1
                centerness = torch.sqrt(
                    (lr_min / (lr_max + 1e-8)) * (tb_min / (tb_max + 1e-8))
                ).clamp(0.0, 1.0)

                ctr_t[pos_mask] = centerness.to(ctr_t.dtype)
                

            cls_t_list.append(cls_t)
            reg_t_list.append(reg_t)
            ctr_t_list.append(ctr_t)
            pos_mask_list.append(pos_mask)

        # ---- 3) Stack targets & masks ----
        cls_tgt = torch.stack(cls_t_list, dim=0)          # [B, L, C]
        reg_tgt = torch.stack(reg_t_list, dim=0)          # [B, L, 4] (normalized by stride)
        ctr_tgt = torch.stack(ctr_t_list, dim=0)          # [B, L]
        pos_mask_bt = torch.stack(pos_mask_list, dim=0)   # [B, L]
        normalizer = max(total_pos, 1)

        # ---- 4) Classification loss (sigmoid focal over all points) -expand---
        # Check for any invalid values in classification targets
        if torch.any(torch.isnan(cls_tgt)) or torch.any(torch.isinf(cls_tgt)):
            print("Warning: Invalid values detected in cls_tgt")
            cls_tgt = torch.nan_to_num(cls_tgt, 0.0)

        # Compute focal loss with alpha and gamma as per FCOS paper
        cls_loss_val = sigmoid_focal_loss(
            cls_flat.reshape(-1, C), 
            cls_tgt.reshape(-1, C),
            alpha=0.25,  # as per FCOS paper
            gamma=2.0    # as per FCOS paper
        )
        if cls_loss_val.dim() > 0:
            cls_loss_val = cls_loss_val.sum()
        cls_loss = cls_loss_val / normalizer

        # ---- 5) Regression loss (GIoU on boxes, only positives) ----
        reg_pos = reg_flat.reshape(-1, 4)[pos_mask_bt.reshape(-1)]       # normalized by stride
        gt_pos = reg_tgt.reshape(-1, 4)[pos_mask_bt.reshape(-1)]         # normalized by stride
        if reg_pos.numel() == 0:
            reg_loss = reg_pos.sum()  # zero
        else:
            # convert to pixel-space boxes with point centers
            strides_pos = strides_all.unsqueeze(0).expand(B, -1).reshape(-1)[pos_mask_bt.reshape(-1)]
            pts_rep = points_all.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2)[pos_mask_bt.reshape(-1)]
            reg_pos_px = reg_pos * strides_pos.unsqueeze(1)
            gt_pos_px = gt_pos * strides_pos.unsqueeze(1)

            pred_boxes = torch.stack([
                pts_rep[:, 0] - reg_pos_px[:, 0],
                pts_rep[:, 1] - reg_pos_px[:, 1],
                pts_rep[:, 0] + reg_pos_px[:, 2],
                pts_rep[:, 1] + reg_pos_px[:, 3],
            ], dim=1)
            tgt_boxes = torch.stack([
                pts_rep[:, 0] - gt_pos_px[:, 0],
                pts_rep[:, 1] - gt_pos_px[:, 1],
                pts_rep[:, 0] + gt_pos_px[:, 2],
                pts_rep[:, 1] + gt_pos_px[:, 3],
            ], dim=1)

            reg_loss_val = giou_loss(pred_boxes, tgt_boxes)
            if reg_loss_val.dim() > 0:
                reg_loss_val = reg_loss_val.sum()
            reg_loss = reg_loss_val / normalizer

        # ---- 6) Centerness loss (BCE-with-logits, only positives) ----
        ctr_pos_logits = ctr_flat.reshape(-1)[pos_mask_bt.reshape(-1)]
        ctr_pos_tgt = ctr_tgt.reshape(-1)[pos_mask_bt.reshape(-1)]
        if ctr_pos_logits.numel() == 0:
            ctr_loss = ctr_pos_logits.sum()  # zero
        else:
            # clamp logits to avoid overflow
            # ctr_pos_logits = ctr_pos_logits.clamp(-8, 8) 
            ctr_loss_val = nn.functional.binary_cross_entropy_with_logits(
                ctr_pos_logits, ctr_pos_tgt, reduction="sum"
            )
            ctr_loss = ctr_loss_val / normalizer

        final_loss = cls_loss + reg_loss + 2 * ctr_loss

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        # points: (5,H,W,2) list
        # strides: (5,) tensor
        # cls_logits: (5, N,C,H,W) list
        # reg_outputs: (5, N,4,H,W) list
        # ctr_logits: (5, N,1,H,W) list
        # image_shapes: (N, H,W) list
        device = cls_logits[0].device
        N = cls_logits[0].shape[0]
        Lvl = len(cls_logits)
        C = self.num_classes

        detections = []

        # helper: clip & valid
        def _clip_and_valid(boxes, img_h, img_w, min_size=0.0):
            boxes[:, 0].clamp_(0, img_w)
            boxes[:, 2].clamp_(0, img_w)
            boxes[:, 1].clamp_(0, img_h)
            boxes[:, 3].clamp_(0, img_h)
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            valid = (ws > 0) & (hs > 0)
            if min_size > 0:
                valid = valid & (ws >= min_size) & (hs >= min_size)
            return boxes, valid

        for n in range(N):
            boxes_all, scores_all, labels_all = [], [], []

            for lvl in range(Lvl):
                cls_lvl = cls_logits[lvl][n]    # [C,H,W]
                ctr_lvl = ctr_logits[lvl][n]    # [1,H,W]
                reg_lvl = reg_outputs[lvl][n]   # [4,H,W]
                H, W = reg_lvl.shape[-2], reg_lvl.shape[-1]
                L = H * W
                if L == 0:
                    continue

                # points (y, x)
                pts = points[lvl].reshape(-1, 2).to(device)      # [L,2] -> (y, x)
                cls_prob = cls_lvl.sigmoid().reshape(C, L)       # [C,L]
                ctr_prob = ctr_lvl.sigmoid().reshape(1, L)       # [1,L]
                scores   = (cls_prob * ctr_prob)                 # [C,L]

                loc_max, _ = scores.max(dim=0)                   # [L]
                loc_keep = (loc_max > self.score_thresh)         # [L] bool
                if not loc_keep.any():
                    continue
                loc_idx_all = torch.nonzero(loc_keep, as_tuple=False).squeeze(1)  # [M_loc]

                reg_flat_all = reg_lvl.reshape(4, L)             # [4,L]
                s = strides[lvl]

                # loc_keep top-k
                for c in range(C):
                    cls_scores_c = scores[c, loc_keep]           # [M_loc]
                    if cls_scores_c.numel() == 0:
                        continue

                    if getattr(self, "topk_candidates", None) and self.topk_candidates > 0:
                        k = min(self.topk_candidates, cls_scores_c.numel())
                        topk_scores_c, topk_pos_local = torch.topk(cls_scores_c, k, sorted=False)  # [k]
                    else:
                        topk_scores_c = cls_scores_c
                        topk_pos_local = torch.arange(cls_scores_c.numel(), device=cls_scores_c.device)

                    keep2 = (topk_scores_c > self.score_thresh)
                    if not keep2.any():
                        continue

                    topk_scores_c = topk_scores_c[keep2]         # [k']
                    topk_pos_local = topk_pos_local[keep2]       # [k']
                    sel_l = loc_idx_all[topk_pos_local]          # [k'] ∈ [0..L)

                    sel_pts  = pts[sel_l]                        # [k',2]  (y, x)
                    reg_flat = reg_flat_all[:, sel_l]            # [4,k']

                    l = reg_flat[0] * s
                    t = reg_flat[1] * s
                    r = reg_flat[2] * s
                    b = reg_flat[3] * s

                    y = sel_pts[:, 0]
                    x = sel_pts[:, 1]
                    x1 = x - l; y1 = y - t
                    x2 = x + r; y2 = y + b

                    boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [k',4]
                    img_h, img_w = image_shapes[n]
                    boxes, valid = _clip_and_valid(boxes, img_h, img_w, min_size=0.1)

                    if valid.any():
                        boxes_all.append(boxes[valid])
                        scores_all.append(topk_scores_c[valid])
                        labels_all.append(torch.full_like(
                            topk_scores_c[valid], c + 1, dtype=torch.int64
                        ))

            if len(boxes_all) == 0:
                detections.append(
                    {
                        "boxes":  torch.zeros((0, 4), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                    }
                )
                continue

            boxes_all  = torch.cat(boxes_all,  dim=0)
            scores_all = torch.cat(scores_all, dim=0)
            labels_all = torch.cat(labels_all, dim=0)

            # NMS
            keep = batched_nms(boxes_all, scores_all, labels_all, self.nms_thresh)
            # sort by scores
            keep = keep[scores_all[keep].argsort(descending=True)]
            if keep.numel() > self.detections_per_img:
                keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes":  boxes_all[keep],
                    "scores": scores_all[keep],
                    "labels": labels_all[keep],
                }
            )

        return detections
