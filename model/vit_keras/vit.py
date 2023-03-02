import typing
import warnings
import tensorflow as tf
import typing_extensions as tx

from model.vit_keras.patch_encoder import PatchEncoder
from . import layers, utils

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)
CONFIG_Ti: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 192,
}
CONFIG_Ti_12_2: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 384,
}
CONFIG_Ti_12_3: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 768,
}
CONFIG_Ti_12_1024: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 1024,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 192,
}
CONFIG_Ti_12_2048: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 2048,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 192,
}
CONFIG_Ti_3: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 3,
    "hidden_size": 192,
}

CONFIG_Ti_4: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 4,
    "hidden_size": 192,
}

CONFIG_Ti_6: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 1024,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_size": 256,
}

CONFIG_Ti_6_2: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_size": 256,
}
CONFIG_Ti_6_3: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 2048,
    "num_heads": 8,
    "num_layers": 6,
    "hidden_size": 256,
}

CONFIG_S: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 1664,
    "num_heads": 6,
    "num_layers": 12,
    "hidden_size": 384,
}

CONFIG_B: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 768,
}

BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
WEIGHTS = {"imagenet21k": 21_843, "imagenet21k+imagenet2012": 1_000}
SIZES = {"B_16", "B_32", "L_16", "L_32"}

ImageSizeArg = typing.Union[typing.Tuple[int, int], int]


def build_model(
    input_shape: tuple,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    x = tf.keras.layers.Input(shape=input_shape)
    proj = tf.keras.layers.Dense(units=hidden_size)(x)
    y = PatchEncoder(input_shape[0], hidden_size)(x)
    y = y + proj
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)
    if include_top:
        y = tf.keras.layers.Dense(classes, name="head", activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def validate_pretrained_top(
    include_top: bool, pretrained: bool, classes: int, weights: str
):
    """Validate that the pretrained weight configuration makes sense."""
    assert weights in WEIGHTS, f"Unexpected weights: {weights}."
    expected_classes = WEIGHTS[weights]
    if classes != expected_classes:
        warnings.warn(
            f"Can only use pretrained_top with {weights} if classes = {expected_classes}. Setting manually.",
            UserWarning,
        )
    assert include_top, "Can only use pretrained_top with include_top."
    assert pretrained, "Can only use pretrained_top with pretrained."
    return expected_classes


def load_pretrained(
    size: str,
    weights: str,
    model: tf.keras.models.Model,
):
    fname = f"ViT-{size}_{weights}.npz"
    origin = f"{BASE_URL}/{fname}"
    local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
    utils.load_weights_numpy(
        model=model,
        params_path=local_filepath
    )


def vit_base(
    input_shape = (10,45),
    classes=2,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )

    if pretrained:
        load_pretrained(
            size="B_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            image_size=input_shape,
            patch_size=16,
        )
    return model

def vit_tiny(
    input_shape = (10,45),
    classes=2,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_custom(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
        num_heads=3,
        mlp_dim=768,
        num_layers=12,
        hidden_size=192
):
    CONFIG_Ti_CUSTOM: ConfigDict = {
        "dropout": 0.1,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }
    model = build_model(
        **CONFIG_Ti_CUSTOM,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_12_2(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_12_2,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_tiny_12_3(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_12_3,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_tiny_12_1024(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_12_1024,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_12_2048(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_12_2048,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_3(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_3,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_4(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_4,
        name="vit-ti",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_6(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_6,
        name="vit-ti_6",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model

def vit_tiny_6_2(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_6_2,
        name="vit-ti_9",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model
def vit_tiny_6_3(
        input_shape = (10,45),
        classes=2,
        activation="linear",
        include_top=True,
        pretrained=True,
        pretrained_top=True,
        weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_Ti_6_3,
        name="vit-ti_6_3",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    return model


def vit_small(
    input_shape = (10,45),
    classes=2,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = build_model(
        **CONFIG_S,
        name="vit-small",
        input_shape=input_shape,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )

    # if pretrained:
    #     load_pretrained(
    #         size="B_16",
    #         weights=weights,
    #         model=model,
    #         pretrained_top=pretrained_top,
    #         image_size=input_shape,
    #         patch_size=16,
    #     )
    return model
#
#
# def build_model_patch_segment(
#     input_shape: tuple,
#     num_layers: int,
#     hidden_size: int,
#     num_heads: int,
#     name: str,
#     window_size: int,
#     mlp_dim: int,
#     dropout=0.1,
#     activation="linear",
#     include_top=True,
#     representation_size=None,
# ):
#     x = tf.keras.layers.Input(shape=input_shape)
#     proj = tf.keras.layers.Dense(units=hidden_size)(x)
#     y = PatchEncoder(input_shape[0], hidden_size)(x)
#     y = y + proj
#     for n in range(num_layers):
#         y, _ = layers.TransformerBlock(
#             num_heads=num_heads,
#             mlp_dim=mlp_dim,
#             dropout=dropout,
#             name=f"Transformer/encoderblock_{n}",
#         )(y)
#     y = tf.keras.layers.LayerNormalization(
#         epsilon=1e-6, name="Transformer/encoder_norm"
#     )(y)
#     if representation_size is not None:
#         y = tf.keras.layers.Dense(
#             representation_size, name="pre_logits", activation="tanh"
#         )(y)
#     if include_top:
#         y = tf.keras.layers.Dense(pow(window_size,2), name="head", activation=activation)(y)
#     return tf.keras.models.Model(inputs=x, outputs=y, name=name)
#
#
#
# def vit_small_patch_segment(
#     input_shape = (10,45),
#     activation="linear",
#     include_top=True,
#     window_size=3,
#     pretrained=True,
#     pretrained_top=True,
#     weights="imagenet21k+imagenet2012",
# ):
#     model = build_model_patch_segment(
#         **CONFIG_S,
#         name="vit-small",
#         input_shape=input_shape,
#         window_size=window_size,
#         activation=activation,
#         include_top=include_top,
#         representation_size=768 if weights == "imagenet21k" else None,
#     )
#
#     # if pretrained:
#     #     load_pretrained(
#     #         size="B_16",
#     #         weights=weights,
#     #         model=model,
#     #         pretrained_top=pretrained_top,
#     #         image_size=input_shape,
#     #         patch_size=16,
#     #     )
#     return model