import segmentation_models_pytorch as smp


def build_model(architecture: str, encoder: str):
    """
    Build a binary segmentation model.

    Output:
        1 channel mask.
        Raw logits, not sigmoid probabilities.
    """
    architecture = architecture.lower()

    if architecture == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    elif architecture == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model