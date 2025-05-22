import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm  # pip install timm

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class HRNetFeatureExtractor(nn.Module):
    def __init__(self, hrnet_model):
        """
        Wraps an HRNet-w18 backbone that outputs a list of feature maps.

        Args:
            hrnet_model (nn.Module): An instance of HRNet-w18 or a similar model that contains
                                     conv1, bn1, act1, conv2, bn2, act2, layer1, transition1, stage2,
                                     transition2, stage3, transition3, and stage4 modules.
        """
        super(HRNetFeatureExtractor, self).__init__()
        self.hrnet = hrnet_model

    def forward(self, x):
        # Process x through the initial layers
        x = self.hrnet.conv1(x)
        x = self.hrnet.bn1(x)
        x = self.hrnet.act1(x)
        x = self.hrnet.conv2(x)
        x = self.hrnet.bn2(x)
        x = self.hrnet.act2(x)
        x = self.hrnet.layer1(x)

        # Transition to multiple resolutions: build the multi‐branch feature maps.
        f1 = x
        m = []
        for module in self.hrnet.transition1:  # Assumes transition1 is a ModuleList
            m.append(module(f1))

        # Stage 2: returns features at two resolutions.
        stage2_features = self.hrnet.stage2(m)

        # Prepare inputs for Stage 3.
        f2 = stage2_features
        m = []  # List to store outputs for transition2
        for i, module in enumerate(self.hrnet.transition2):
            # Use the corresponding feature if within range; otherwise, replicate the last feature.
            feature = f2[i] if i < len(f2) else f2[-1]
            if isinstance(module, nn.Identity):
                m.append(feature)
            else:
                m.append(module(feature))

        # Stage 3: returns features at three resolutions.
        stage3_features = self.hrnet.stage3(m)

        # Prepare inputs for Stage 4.
        f3 = stage3_features
        m = []  # List to store outputs for transition3
        for i, module in enumerate(self.hrnet.transition3):
            # Use the corresponding feature if available; otherwise, replicate the last feature.
            feature = f3[i] if i < len(f3) else f3[-1]
            if isinstance(module, nn.Identity):
                m.append(feature)
            else:
                m.append(module(feature))

        # Stage 4: returns features at four resolutions.
        stage4_features = self.hrnet.stage4(m)

        return stage4_features

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        """
        Channel attention uses both average pooling and max pooling operations to aggregate spatial
        information and then learns channel-wise attention.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        Spatial attention computes an attention map along the spatial dimension. It uses both average and max along
        the channel dimension to feed a small convolution.
        """
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(x_cat))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Combines channel and spatial attention.
        """
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, ratio=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x_out = self.channel_att(x)
        x_out = self.spatial_att(x_out)
        return x_out

###########################################################
# HRNet Backbone with CBAM
###########################################################
class HRNetBackbone(nn.Module):
    def __init__(self, use_cbam=True):
        """
        Loads a pretrained HRNet-w18. Using timm’s features_only mode returns multi-scale features.
        The HRNet model from timm will return a list of feature maps for stages specified via out_indices.
        """
        super(HRNetBackbone, self).__init__()
        # Create HRNet-w18 with multi-scale feature outputs.
        # out_indices specifies which stage outputs to return.
        # self.hrnet = timm.create_model('hrnet_w18', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        model = timm.create_model('hrnet_w18', pretrained=False, num_classes = 2)
        
        # Construct relative path to the checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "saves", "checkpoints")
        checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, "best_seg.pth"))

        if os.path.exists(checkpoint_path):
            print(f"✅ Loading HRNet weights from: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}. Skipping HRNet weight loading.")
        self.hrnet = HRNetFeatureExtractor(model)
        self.use_cbam = use_cbam

        # Get the channels for each output feature map
        # self.channels_list = [self.hrnet.feature_info.channels()]
        self.channels_list = [18, 36, 72, 144]

        # Create a CBAM module for each scale if enabled.
        if self.use_cbam:
            self.cbam_modules = nn.ModuleList([CBAM(ch) for ch in self.channels_list])
        else:
            self.cbam_modules = None

    def forward(self, x):
        features = self.hrnet(x)
        if self.use_cbam and self.cbam_modules is not None:
            features = [cbam(feature) for feature, cbam in zip(features, self.cbam_modules)]
        return features  # List of feature maps


###########################################################
# Combine HRNet features with FPN for Detection & Segmentation
###########################################################
class HRNetFPNBackbone(nn.Module):
    def __init__(self, out_channels=256, use_cbam=True):
        """
        This backbone wraps HRNet (with optional CBAM) and then fuses the multi
        scale features into a Feature Pyramid (FPN), which is suitable for a Mask R-CNN model.
        """
        super(HRNetFPNBackbone, self).__init__()
        self.hrnet_backbone = HRNetBackbone(use_cbam=use_cbam)
        # Use the channel counts produced by HRNet for the FPN input
        in_channels_list = self.hrnet_backbone.channels_list

        # Create an FPN that takes in the multi-scale features and produces out_channels at each level.
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        # Get feature maps from HRNet (possibly modified by CBAM)
        features = self.hrnet_backbone(x)
        # Create an OrderedDict required by the FPN.
        fpn_in = OrderedDict()
        for idx, feature in enumerate(features):
            # Naming convention: string keys such as '0', '1', etc.
            fpn_in[str(idx)] = feature

        # Fuse the features using the FPN.
        fpn_out = self.fpn(fpn_in)
        return fpn_out


###########################################################
# Assemble the Full Mask R-CNN Model with Custom Backbone
###########################################################
def get_maskrcnn_model(num_classes, backbone_out_channels=256, use_cbam=True):
    """
    This function creates a Mask R-CNN model with the HRNet-FPN backbone.
    The model performs box detection and mask segmentation.
    """
    # Create our custom backbone with FPN
    backbone = HRNetFPNBackbone(out_channels=backbone_out_channels, use_cbam=use_cbam)
    # NOTE:
    # MaskRCNN expects the backbone module to have an attribute .out_channels.
    backbone.out_channels = backbone_out_channels

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),  # One tuple per feature map
        aspect_ratios=((0.5, 1.0, 2.0),) * 4     # Repeat the same aspect ratios for 4 levels
    )


    # Define RoI pooling modules on top of the feature maps.
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    # Create the Mask R-CNN model using our custom backbone.
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,  # set according to your dataset (including background if needed)
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    return model