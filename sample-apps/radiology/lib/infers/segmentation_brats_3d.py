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
    GaussianSmoothd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CenterSpatialCropd
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class Segmentation(BasicInferTask):
    """
    This provides Inference Engine for pre-trained Segmentation (SegResNet) model.
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.0, 1.0, 1.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="A pre-trained model for volumetric (3D) Segmentation from CT image",
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
        channels = data.get("input_channels", 1)
        t = [
            (
                LoadImaged(keys="image", reader="ITKReader", ensure_channel_first=True)
                if data.get("multi_file", False) is False
                else LoadDirectoryImagesd(keys="image", target_spacing=self.target_spacing, channels=channels)
            ),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image", channel_dim=0),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=self.target_spacing, allow_missing_keys=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=2.0,
                upper=98.0,
                b_min=-1.0,
                b_max=1.0,
                clip=False,
                relative = False,
                channel_wise=True
            ),
            CenterSpatialCropd(
                keys=["image"],
                roi_size=[self.roi_size[0], self.roi_size[1], self.roi_size[2]],
            ),
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
        t = [
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ]

        if data and data.get("largest_cc", False):
            t.append(KeepLargestConnectedComponentd(keys="pred"))
        t.extend(
            [
                Restored(
                    keys="pred",
                    ref_image="image",
                    config_labels=self.labels if data.get("restore_label_idx", False) else None,
                    # invert_orient=True,
                ),
                GetCentroidsd(keys="pred", centroids_key="centroids"),
            ]
        )
        return t
