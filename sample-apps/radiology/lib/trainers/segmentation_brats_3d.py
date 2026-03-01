# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from lib.transforms.transforms import LoadDirectoryImagesd, NormalizeLabelsInDatasetd
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        roi_size=(224, 224, 144),
        target_spacing=(1.0, 1.0, 1.0),
        num_samples=4,
        description="Train BraTS Segmentation model (TC/WT/ET multilabel)",
        **kwargs,
    ):
        self._network = network
        self.roi_size = roi_size
        self.target_spacing = target_spacing
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=1e-4, weight_decay=1e-5)

    def loss_function(self, context: Context):
        # BraTS is a sigmoid multilabel task (TC, WT, ET).
        # to_onehot_y=False because the label is already 3-channel after
        # ConvertToMultiChannelBasedOnBratsClassesd.
        # sigmoid=True because each channel is independent (not mutually exclusive).
        return DiceLoss(
            smooth_nr=0,
            smooth_dr=1e-5,
            squared_pred=True,
            to_onehot_y=False,
            sigmoid=True,
        )

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
        """
        Transforms follow the official MONAI BraTS tutorial exactly.

        Two loading paths:
          - multi_file=False : image is already a single 4-channel .nii.gz volume
                               (LoadImaged handles it, then EnsureChannelFirstd is a no-op
                                because ITKReader + ensure_channel_first already adds the channel dim)
          - multi_file=True  : a directory of 4 single-modality files is stacked by
                               LoadDirectoryImagesd into a (4, H, W, D) tensor
        """
        channels = context.input_channels
        # multi_file = strtobool(str(context.multi_file))  # ← coerce safely
        # logger.info(f"\n\n Usingh {channels} and multi_file: {multi_file}\n\n")
        # print(f"\n\n Usingh {channels} and multi_file: {multi_file}\n\n")

        return [
            LoadDirectoryImagesd(keys="image", target_spacing=self.target_spacing, channels=channels),
            LoadImaged(keys="label", reader="ITKReader", ensure_channel_first=True),
            # (
            #     LoadImaged(keys="image", reader="ITKReader", ensure_channel_first=True)
            #     if context.multi_file is False
            #     else LoadDirectoryImagesd(keys="image", target_spacing=self.target_spacing, channels=channels)
            # ),
            # ConvertToMultiChannelBasedOnBratsClassesd converts the integer label map
            # to a 3-channel binary tensor: [TC, WT, ET].
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # Random crop matching the official tutorial roi
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=self.roi_size,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Channel-wise zero-mean / unit-std normalisation on non-zero voxels only
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]

    def train_post_transforms(self, context: Context):
        """
        Post-transforms for TRAINING metrics.

        Because this is a sigmoid multilabel task:
          - Apply sigmoid activation per channel.
          - Threshold at 0.5 to get binary predictions.
          - The label is already binary 3-channel — no argmax / to_onehot needed.
        """
        return [
            EnsureTyped(keys="pred", device=context.device),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
            # label is already binary 3-channel, nothing to do
        ]

    def val_pre_transforms(self, context: Context):
        channels = context.input_channels
        # multi_file = strtobool(str(context.multi_file))  # ← same fix
        # logger.info(f"\n\n Usingh {channels} and multi_file: {multi_file}\n\n")
        #
        # print(f"\n\n Usingh {channels} and multi_file: {multi_file}\n\n")
        return [
            LoadDirectoryImagesd(keys="image", target_spacing=self.target_spacing, channels=channels),
            LoadImaged(keys="label", reader="ITKReader", ensure_channel_first=True),
            # (
            #     LoadImaged(keys="image", reader="ITKReader", ensure_channel_first=True)
            #     if context.multi_file is False
            #     else LoadDirectoryImagesd(keys="image", target_spacing=self.target_spacing, channels=channels)
            # ),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # No crop during validation — sliding window covers the full volume
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=2,
            overlap=0.4,
            padding_mode="replicate",
            mode="gaussian",
        )

    def norm_labels(self):
        """Return label dict suitable for region_wise_metrics (1-indexed, no background)."""
        new_label_nums = {}
        for idx, key_label in enumerate(self._labels.keys(), start=1):
            if key_label != "background":
                new_label_nums[key_label] = idx
        return new_label_nums

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "train_mean_dice", "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self.norm_labels(), "val_mean_dice", "val")

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(
                    log_dir=context.events_dir,
                    batch_transform=from_engine(["image", "label"]),
                    output_transform=from_engine(["pred"]),
                    interval=20,
                    epoch_level=True,
                )
            )
        return handlers
