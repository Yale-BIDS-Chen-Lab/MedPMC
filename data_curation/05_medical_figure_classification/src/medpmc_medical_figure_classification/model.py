"""Released ViT checkpoint loading and inference for Stage 5."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Yale-BIDS-Chen/medpmc-med-fig-classification-vit"
DEFAULT_CHECKPOINT_FILENAME = "model.pth.tar"
CLASS_NAMES = {0: "non-medical", 1: "medical"}


def resolve_device(requested: str) -> str:
    import torch

    value = str(requested).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested with --device {requested}, but torch.cuda.is_available() is False"
        )
    return requested


def torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_classifier(
    model_name: str,
    checkpoint_filename: str,
    device: str,
):
    try:
        import timm
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install with: pip install -e '.[inference]'"
        ) from exc

    checkpoint_path = hf_hub_download(repo_id=model_name, filename=checkpoint_filename)
    checkpoint = torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict) or "arch" not in checkpoint or "state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} must contain 'arch' and 'state_dict'"
        )

    architecture = str(checkpoint["arch"])
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    head_weight = state_dict.get("head.weight")
    if head_weight is None or int(head_weight.shape[0]) != 2:
        raise RuntimeError("The released classifier must have a two-class head")

    # The released checkpoint contains the complete model state, so no
    # pretrained base weights are loaded before the strict state-dict load.
    model = timm.create_model(architecture, pretrained=False, num_classes=2)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, architecture, Path(checkpoint_path), checkpoint


def checkpoint_image_size(model) -> int:
    patch_embed = getattr(model, "patch_embed", None)
    value = getattr(patch_embed, "img_size", None)
    if value is None:
        config = getattr(model, "pretrained_cfg", None) or {}
        input_size = config.get("input_size")
        if input_size and len(input_size) >= 3:
            value = input_size[-2:]
    if isinstance(value, int):
        height = width = int(value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise RuntimeError("Could not determine the checkpoint input size")
    if height != width:
        raise RuntimeError(f"Non-square model input is unsupported: {height}x{width}")
    return height


def resolve_timm_data_config(model, requested_image_size: int | None = None) -> dict[str, Any]:
    """Resolve evaluation preprocessing from the loaded timm model.

    With no explicit transform overrides, timm provides the model input size,
    crop, interpolation, mean, and standard deviation used by the
    manifest-backed dataset.
    """
    from timm.data import resolve_data_config

    expected_size = checkpoint_image_size(model)
    overrides: dict[str, Any] = {}
    if requested_image_size is not None:
        if int(requested_image_size) != expected_size:
            raise ValueError(
                f"--image-size {requested_image_size} does not match the checkpoint "
                f"input size {expected_size}x{expected_size}"
            )
        overrides["input_size"] = (3, expected_size, expected_size)
    config = dict(resolve_data_config(overrides, model=model))
    input_size = tuple(int(value) for value in config.get("input_size", ()))
    if len(input_size) != 3 or input_size[-2:] != (expected_size, expected_size):
        raise RuntimeError(
            f"Resolved timm input_size {input_size!r} disagrees with checkpoint size {expected_size}"
        )
    config["input_size"] = input_size
    return config


def build_preprocess(data_config: dict[str, Any]):
    from timm.data import create_transform

    return create_transform(**data_config, is_training=False)


def jsonable_data_config(data_config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("input_size", "interpolation", "mean", "std", "crop_pct", "crop_mode"):
        value = data_config.get(key)
        if isinstance(value, tuple):
            value = list(value)
        output[key] = value
    return output


def predict_probabilities(model, inputs, *, device: str, amp: bool):
    import torch

    inputs = inputs.to(device, non_blocking=device.startswith("cuda"))
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.detach().cpu()
