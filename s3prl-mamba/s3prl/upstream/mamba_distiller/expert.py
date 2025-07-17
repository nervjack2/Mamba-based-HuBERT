"""
    Upstream expert for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import torch
import yaml

from ..interfaces import UpstreamBase
from .builder import PretrainedDistiller


class UpstreamExpert(UpstreamBase):
    """
    The Distiller wrapper
    """

    def __init__(self, ckpt, model_config=None, **kwargs):
        super().__init__(**kwargs)

        if model_config is not None:
            print(
                "[UpstreamExpert] - Using upstream expert config file from:",
                model_config,
            )
            with open(model_config, "r") as file:
                options = yaml.load(file, Loader=yaml.FullLoader)
        else:
            print("[UpstreamExpert] - Using the default upstream expert config")
            options = {
                "load_pretrain": "True",
                "no_grad": "False",
                "permute_input": "False",
            }

        options["ckpt_file"] = ckpt

        self.model = PretrainedDistiller(options)

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs, no_pred=False):
        _, feat_final, pred, pad_mask, layer_hidden = self.model(
            wavs, get_hidden=True, no_pred=no_pred
        )

        hidden_feats = [feat_final] + layer_hidden

        states = {
            "hidden_states": hidden_feats
        }

        return states
