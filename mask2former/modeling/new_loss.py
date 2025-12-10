import torch
import torch.nn as nn
import torch.nn.functional as F

class OPPLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, floss_beta=0.3, floss_log_like=False):
        super(OPPLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.floss_beta = floss_beta
        self.floss_log_like = floss_log_like
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def bce_dice_loss(self, inputs, targets):
        # inputs: [batch_size, H, W], targets: [batch_size, 1, H, W]; inputs expected as logits
        bce = self.bce_loss_fn(inputs, targets)
        inter = (inputs * targets).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        dice_loss = 1 - dice.mean()
        return bce + dice_loss

    def cross_entropy_loss(self, inputs, targets):
        # inputs: [batch_size, 12, H, W], targets: [batch_size, 1, H, W]
        loss = 0
        targets = targets.unsqueeze(1)
        for i in range(12):
            loss += self.bce_loss_fn(inputs[:, i:i+1, :, :], targets)
        return loss / 12  # average across 12 channels

    def f_loss(self, prediction, target):
        # prediction: [batch_size, H, W], target: [batch_size, 1, H, W]
        EPS = 1e-10
        target = target.squeeze(1)  # [batch_size, H, W]
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.floss_beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.floss_beta) * TP / (H + EPS)
        if self.floss_log_like:
            loss = -torch.log(fmeasure)
        else:
            loss = 1 - fmeasure
        return loss.mean()

    def forward(self, inputs, targets):
        # inputs: [batch_size, 12, H, W] logits; targets: List[Dict] with key "opp_gt" of shape [1, H, W]
        total_loss = 0
        for i, target in enumerate(targets):
            opp_gt = target["opp_gt"]  # [1, H, W]
            input_i = inputs[i:i+1]  # [1, 12, H, W]

            if input_i.shape[-2:] != opp_gt.shape[-2:]:
                input_i = F.interpolate(
                    input_i,
                    size=opp_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # compute BCE+Dice over 12 channels and average
            bce_dice_losses = sum([
                self.bce_dice_loss(input_i[:, j, :, :], opp_gt)
                for j in range(12)
            ]) / 12  # average BCE+Dice loss

            ce_loss = self.cross_entropy_loss(input_i, opp_gt)

            # compute F-measure-like loss over 12 channels and average
            f_losses = sum([
                self.f_loss(input_i[:, j, :, :], opp_gt)
                for j in range(12)
            ]) / 12  # average F-measure loss

            # combine losses
            total_loss += (
                self.bce_weight * bce_dice_losses +
                ce_loss +
                self.dice_weight * f_losses
            )

        return total_loss / len(targets)  # average over batch