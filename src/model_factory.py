from src.model import ResNet18CAM
from src.model_with_adapter import ResNet18CAMWithAdapter


MODEL_REGISTRY = {
    "plain": ResNet18CAM,
    "adapter": ResNet18CAMWithAdapter,
}


def build_model(config: dict, num_classes: int):
    model_type = str(config.get("model_type", "plain")).lower()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_type]

    if model_type == "plain":
        return model_cls(
            num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)),
            use_adapter=bool(config.get("use_adapter", False)),
            freeze_backbone=bool(config.get("freeze_backbone", False)),
            adapter_reduction=int(config.get("adapter_reduction", 4)),
        )

    if model_type == "adapter":
        return model_cls(
            num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)),
            gamma=int(config.get("adapter_gamma", 4)),
            adapter_kernel_size=int(config.get("adapter_kernel_size", 3)),
            freeze_backbone=bool(config.get("freeze_backbone", True)),
        )

    raise RuntimeError("unreachable")