"""Protocol-frozen LPIPS v0.1 AlexNet inference without package installation.

This is a narrow, evaluation-only reimplementation of the LPIPS 0.1.4 Alex
forward pass.  It follows the upstream implementation at commit
c33f89e9f46522a584cf41d8880eb0afa982708b and requires both weight files to
be supplied explicitly.  It never downloads weights.
"""
from __future__ import annotations


LPIPS_UPSTREAM_COMMIT = "c33f89e9f46522a584cf41d8880eb0afa982708b"
LPIPS_IMPLEMENTATION_VERSION = "treering-lpips-alex-v0.1"


def _load_weights(torch, path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older torch releases.
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and set(value) == {"state_dict"}:
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise ValueError(f"Expected a state dictionary in {path}")
    return value


def create_frozen_lpips_alex(linear_checkpoint, alexnet_checkpoint):
    """Create the frozen LPIPS Alex model from two explicit local files."""
    import torch
    from torch import nn
    from torchvision.models import alexnet

    class NetLinLayer(nn.Module):
        def __init__(self, channels):
            super().__init__()
            # The dropout module preserves the official v0.1 checkpoint key
            # layout.  Evaluation mode makes it an identity operation.
            self.model = nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(channels, 1, 1, stride=1, padding=0, bias=False),
            )

        def forward(self, value):
            return self.model(value)

    class FrozenLpipsAlex(nn.Module):
        def __init__(self):
            super().__init__()
            trunk = alexnet(weights=None)
            trunk_state = _load_weights(torch, alexnet_checkpoint)
            incompatible = trunk.load_state_dict(trunk_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError("AlexNet checkpoint is incompatible")

            layers = list(trunk.features.children())
            self.slice1 = nn.Sequential(*layers[0:2])
            self.slice2 = nn.Sequential(*layers[2:5])
            self.slice3 = nn.Sequential(*layers[5:8])
            self.slice4 = nn.Sequential(*layers[8:10])
            self.slice5 = nn.Sequential(*layers[10:12])
            for parameter in self.parameters():
                parameter.requires_grad = False

            self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
            self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])
            channels = (64, 192, 384, 256, 256)
            for index, count in enumerate(channels):
                setattr(self, f"lin{index}", NetLinLayer(count))

            linear_state = _load_weights(torch, linear_checkpoint)
            expected = {f"lin{index}.model.1.weight" for index in range(5)}
            if set(linear_state) != expected:
                raise ValueError("LPIPS linear checkpoint has unexpected keys")
            for index in range(5):
                key = f"lin{index}.model.1.weight"
                layer = getattr(self, f"lin{index}").model[1]
                weight = linear_state[key]
                if tuple(weight.shape) != tuple(layer.weight.shape):
                    raise ValueError(f"LPIPS linear weight shape mismatch: {key}")
                layer.weight.data.copy_(weight)
            for parameter in self.parameters():
                parameter.requires_grad = False

        @staticmethod
        def _normalize(value, epsilon=1e-10):
            norm = torch.sqrt(torch.sum(value ** 2, dim=1, keepdim=True))
            return value / (norm + epsilon)

        def _features(self, value):
            outputs = []
            for layer in (self.slice1, self.slice2, self.slice3, self.slice4, self.slice5):
                value = layer(value)
                outputs.append(value)
            return outputs

        def forward(self, reference, candidate):
            reference = (reference - self.shift) / self.scale
            candidate = (candidate - self.shift) / self.scale
            reference_features = self._features(reference)
            candidate_features = self._features(candidate)
            result = None
            for index, (first, second) in enumerate(zip(reference_features, candidate_features)):
                difference = (self._normalize(first) - self._normalize(second)) ** 2
                value = getattr(self, f"lin{index}")(difference).mean((2, 3), keepdim=True)
                result = value if result is None else result + value
            return result

    return FrozenLpipsAlex()
