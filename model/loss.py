import torch
from torch import nn


class DiceBCELogitsSmoothingLoss(nn.Module):
    def __init__(self, eps: float = 0.1, smooth: float = 1.0):
        """
        Combined Dice Loss and BCE Loss with label smoothing for binary segmentation tasks.

        Args:
            eps (float): Label smoothing factor (between 0 and 1). Default is 0.1.
            smooth (float): Smoothing constant to avoid division by zero. Default is 1.0.
        """
        super(DiceBCELogitsSmoothingLoss, self).__init__()
        assert 0 <= eps < 1, "eps must be between 0 and 1."
        self.eps = eps
        self.smooth = smooth

        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def _soft_dice_loss(self, y_pred, y_true):
        """
        Calculate the soft Dice coefficient.

        Args:
            y_true (torch.Tensor): Ground truth tensor.
            y_pred (torch.Tensor): Predicted logits tensor.

        Returns:
            float: Dice coefficient.
        """
        # Apply sigmoid to logits for dice computation
        y_pred = torch.sigmoid(y_pred)
        
        y_pred, y_true = y_pred.flatten(), y_true.flatten()
        y_pred_sum, y_true_sum = y_pred.sum(), y_true.sum()
        intersection = torch.sum(y_pred * y_true)
            
        loss = 1 - (2.0 * intersection + self.smooth) / (y_pred_sum + y_true_sum + self.smooth)
        return loss
        
    def forward(self, y_pred, y_true):
        """
        Compute the combined Dice and BCE loss with optional label smoothing.

        Args:
            y_pred (torch.Tensor): Predicted logits tensor.
            y_true (torch.Tensor): Ground truth tensor.

        Returns:
            float: Combined loss value.
        """
        # Dice loss
        dice_loss = self._soft_dice_loss(y_pred, y_true)

        # Smooth BCE loss
        if self.eps > 0:
            smooth_y_true = y_true * (1 - self.eps) + (1 - y_true) * self.eps
        else:
            smooth_y_true = y_true
            
        bce_loss = self.bce_loss(y_pred, smooth_y_true)

        # Combined loss
        return dice_loss + bce_loss
