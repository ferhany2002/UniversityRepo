
import torch


from .accel import (
    SegmentationHead as AccelSegmentationhead,
    ResNetBodyNoChannelPool as AccelModelBodyNoPool)


class SegmentationFusionModel(torch.nn.Module):
    def __init__(self,
                 modalities,
                 mask_len):
        """
        """
        super().__init__()

        self.modalities = modalities

        if 'accel' in modalities:
            self.accel_feature_extractor = AccelModelBodyNoPool(c_in=1)
            self.accel_head = AccelSegmentationhead(c_out=1, output_len=mask_len)

    def forward(self, batch: dict):
        """
        """
        masks = []
        if 'data' in batch:

            f = self.accel_feature_extractor(batch['data'])

            u = self.accel_head(f)


            masks.append(u)

        masks = torch.stack(masks, dim=2)

        masks = masks.mean(dim=2)


        return masks
