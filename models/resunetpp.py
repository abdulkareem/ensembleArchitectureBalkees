import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNetPPWrapper(nn.Module):
    def __init__(self, repo_file: str, out_size: int = 256):
        super().__init__()
        path = Path(repo_file)
        if not path.exists():
            raise FileNotFoundError(f"ResUNet++ source file not found: {path}")

        spec = importlib.util.spec_from_file_location("resunetpp_module", str(path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        self.model = module.build_resunetplusplus()
        self.out_size = out_size
        self._printed_shapes = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.shape[2:] != (self.out_size, self.out_size):
            logits = F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        if not self._printed_shapes:
            print(f"[ResUNet++] input={tuple(x.shape)} output={tuple(logits.shape)}")
            self._printed_shapes = True
        return logits
