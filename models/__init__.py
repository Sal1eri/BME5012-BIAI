"""Model registry."""


def build_model(name: str, num_classes: int = 5, in_channels: int | None = None):
    """Build a model by name."""
    name = name.lower()
    if name in {"resnet", "resnet3d"}:
        from .resnet import ResNet3D

        return ResNet3D(num_classes=num_classes, input_channels=in_channels or 1)
    if name == "vgg16":
        from .vgg16 import VGG16ForAD

        return VGG16ForAD(num_classes=num_classes)
    if name == "hcct_vit":
        from .hcct_vit import HCCT_ViT

        return HCCT_ViT(num_classes=num_classes, input_channels=in_channels or 1)
    if name == "vit_3d":
        from .vit_3d import VisionTransformer3D

        return VisionTransformer3D(num_classes=num_classes, in_chans=in_channels or 1)
    if name == "swin_3d_cnn":
        from .swin_3d_cnn import SwinTransformer3D_CNN

        return SwinTransformer3D_CNN(num_classes=num_classes, in_chans=in_channels or 1)
    if name == "mvst":
        from .mvst import MultiViewSliceTransformer

        num_slices = 12
        return MultiViewSliceTransformer(num_classes=num_classes, num_slices=num_slices)
    raise ValueError(f"Unknown model: {name}")
