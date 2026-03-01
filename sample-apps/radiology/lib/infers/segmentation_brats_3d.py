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

from typing import Callable, Sequence

from lib.transforms.transforms import GetCentroidsd, LoadDirectoryImagesd
from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class Segmentation(BasicInferTask):
    """
    Inference Engine for BraTS brain tumour segmentation using a SegResNet.

    The model outputs 3 channels (TC, WT, ET) with sigmoid activations — it is
    a multilabel task, NOT a softmax classification.  Each channel is thresholded
    independently at 0.5 to produce binary maps.

    Two image loading modes are supported (set via ``data["multi_file"]``):
      - False (default): the input image is a single 4-channel NIfTI volume.
      - True:            ``data["image"]`` is a directory containing 4 single-
                         modality NIfTI files; LoadDirectoryImagesd stacks them.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="Pre-trained BraTS SegResNet — TC/WT/ET multilabel segmentation",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        """
        Pre-processing pipeline matching the official MONAI BraTS tutorial.

        NOTE: ScaleIntensityRangePercentilesd and CenterSpatialCropd from the
        original file have been removed — they are not part of the BraTS pipeline
        and would distort MRI intensity normalisation.  NormalizeIntensityd with
        nonzero=True, channel_wise=True is the correct approach for multi-modal MRI.
        """
        channels = data.get("input_channels", 4)
        t = [
            (
                LoadImaged(keys="image", reader="ITKReader", ensure_channel_first=True)
                if data.get("multi_file", False) is False
                else LoadDirectoryImagesd(
                    keys="image",
                    target_spacing=self.target_spacing,
                    channels=channels,
                )
            ),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            # EnsureChannelFirstd is safe to keep as a guard; if the channel dim is
            # already present (ITKReader + ensure_channel_first) it is a no-op.
            EnsureChannelFirstd(keys="image", channel_dim=0),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(
                keys="image",
                pixdim=self.target_spacing,
                allow_missing_keys=True,
            ),
            # Channel-wise intensity normalisation on non-zero voxels only.
            # This matches both the tutorial and the training pipeline exactly.
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
        return t

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.roi_size,
            sw_batch_size=2,
            overlap=0.4,
            padding_mode="replicate",
            mode="gaussian",
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        """
        Post-processing for multilabel sigmoid output.

        IMPORTANT differences from a softmax segmentation:
          - Activationsd uses sigmoid=True (not softmax=True).
          - AsDiscreted thresholds each channel at 0.5 independently
            (not argmax, because channels are not mutually exclusive).
          - KeepLargestConnectedComponentd is applied per-channel if available.
        """
        t = [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            # Sigmoid: each of the 3 channels (TC, WT, ET) is activated independently.
            Activationsd(keys="pred", sigmoid=True),
            # Threshold each channel at 0.5 to produce binary masks.
            AsDiscreted(keys="pred", threshold=0.5),
        ]

        if data and data.get("largest_cc", False):
            # Apply per-channel so TC, WT and ET are each cleaned independently.
            t.append(
                KeepLargestConnectedComponentd(
                    keys="pred",
                    independent=True,   # treat each channel separately
                )
            )

        t.extend(
            [
                Restored(
                    keys="pred",
                    ref_image="image",
                    config_labels=self.labels if data.get("restore_label_idx", False) else None,
                ),
                GetCentroidsd(keys="pred", centroids_key="centroids"),
            ]
        )
        return t
