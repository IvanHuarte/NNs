import os
import copy
import json
import ast
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy.typing as npt
from flax.serialization import to_bytes, from_bytes
from typing import Optional
from datetime import date
from platform import architecture, python_version
import pathlib
from NN_module.models.MLP import BatchedMultiLayerPerceptron
from NN_module.models.ViT import BatchedSpinViT
from NN_module.models.ViT_2D import BatchedSpinViT_2D
from NN_module.models.CNN import CNN
from NN_module.models.CvT import CvT
from NN_module.models.CvT2 import CvT2
from NN_module.models.CvT3 import CvT3
from NN_module.models.CvTaps import CvTaps
from NN_module.models.split_training import (
    SplitTraining_ViT_MLP,
    SplitTraining_ViT_CNN,
    SplitTraining_CvT_CNN,
)
from NN_module.models.ST_modules.CvT3_CNNPh import CvT3_CNNPh
from NN_module.models.ST_modules.CvT3_EDPPh import CvT3_EDPPh
from NN_module.models.ST_modules.CvT3_CNNClsf import CvT3_CNNClsf
from NN_module.models.ST_modules.CvT3_CvT3 import CvT3_CvT3
from NN_module.models.ST_modules.CvTaps_CvTaps import CvTaps_CvTaps
from NN_module.models.ST_modules.ViT2D_CNN import ViT2D_CNN
from NN_module.models.ST_modules.ViT2D_CNNClsf import ViT2D_CNNClsf

from NN_module.NN_utils import activation_dict, sampler_dict, rule_dict



def init_model(name, model_setup):

    if name == "MLP":
        activation = model_setup["activation"]
        if all([type(act) in [str, int] for act in activation]):
            activation = tuple(
                [activation_dict[act] if act != 0 else 0 for act in activation]
            )

        return BatchedMultiLayerPerceptron(
            lattice_size=tuple(model_setup["lattice_size"]),
            hidden_alpha=tuple(model_setup["hidden_alpha"]),
            activation=activation,
            param_dtype=jnp.complex128,
            output_dim=1,
            symm_2D=model_setup["symm_2D"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )

    elif name == "ViT":
        return BatchedSpinViT(
            token_size=model_setup["token_size"],
            embedding_d=model_setup["embedding_d"],
            n_heads=model_setup["n_heads"],
            n_blocks=model_setup["n_blocks"],
            n_ffn_layers=model_setup["n_ffn_layers"],
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            is_complex=model_setup["is_complex"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )

    elif name == "ViT_2D":
        return BatchedSpinViT_2D(
            lattice_size=tuple(model_setup["lattice_size"]),
            token_size=tuple(model_setup["token_size"]),
            embedding_d=model_setup["embedding_d"],
            n_heads=model_setup["n_heads"],
            n_blocks=model_setup["n_blocks"],
            n_ffn_layers=model_setup["n_ffn_layers"],
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            is_complex=model_setup["is_complex"],
            symm_2D=model_setup["symm_2D"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "CNN":
        return CNN(
            lattice_size=tuple(model_setup["lattice_size"]),
            block_channels=tuple(model_setup["block_channels"]),
            kernel_size=tuple(model_setup["kernel_size"]),
            n_ffn_layers_cnn=model_setup["n_ffn_layers_cnn"],
        )
    elif name == "CvT":
        return CvT(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            two_heads=model_setup["two_heads"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "CvT2":
        return CvT2(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            strides_list=tuple([tuple(st) for st in model_setup["strides"]]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            two_heads=model_setup["two_heads"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )

    elif name == "CvT3":
        return CvT3(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            two_heads=model_setup["two_heads"],
            two_heads_sincos=model_setup["two_heads_sincos"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
            phasors=model_setup["phasors"],
        )

    elif name == "CvTaps":
        return CvTaps(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            channels_list=tuple(model_setup["channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            strides_list=tuple([tuple(st) for st in model_setup["strides"]]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            two_heads=model_setup["two_heads"],
            two_heads_sincos=model_setup["two_heads_sincos"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
            phasors=model_setup["phasors"],
        )

    elif name == "SplitTraining_ViT_MLP":

        activation = model_setup["activation"]
        if all([type(act) in [str, int] for act in activation]):
            activation = tuple(
                [activation_dict[act] if act != 0 else 0 for act in activation]
            )
        return SplitTraining_ViT_MLP(
            lattice_size=tuple(model_setup["lattice_size"]),
            token_size=tuple(model_setup["token_size"]),
            embedding_d=model_setup["embedding_d"],
            n_heads=model_setup["n_heads"],
            n_blocks=model_setup["n_blocks"],
            n_ffn_layers=model_setup["n_ffn_layers"],
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            is_complex=model_setup["is_complex"],
            symm_2D_modulus=model_setup["symm_2D_modulus"],
            symm_Z2_modulus=model_setup["symm_Z2_modulus"],
            trivial_Z2_modulus=model_setup["trivial_Z2_modulus"],
            param_dtype_phase=jnp.float64,
            hidden_alpha=tuple(model_setup["hidden_alpha"]),
            activation=tuple(activation),
            output_dim=model_setup["output_dim"],
            symm_2D_phase=model_setup["symm_2D_phase"],
            symm_Z2_phase=model_setup["symm_Z2_phase"],
            trivial_Z2_phase=model_setup["trivial_Z2_phase"],
        )
    elif name == "ViT2D_CNN":
        return ViT2D_CNN(
            lattice_size=tuple(model_setup["lattice_size"]),
            token_size=tuple(model_setup["token_size"]),
            embedding_d=model_setup["embedding_d"],
            n_heads=model_setup["n_heads"],
            n_blocks=model_setup["n_blocks"],
            n_ffn_layers=model_setup["n_ffn_layers"],
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            is_complex=model_setup["is_complex"],
            block_channels=tuple(model_setup["block_channels"]),
            kernel_size=tuple(model_setup["kernel_size"]),
            n_ffn_layers_cnn=model_setup["n_ffn_layers_cnn"],
            symm_2D=model_setup["symm_2D"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "SplitTraining_CvT_CNN":
        return SplitTraining_CvT_CNN(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            block_channels_cnn=tuple(model_setup["block_channels_cnn"]),
            kernel_size_cnn=tuple(model_setup["kernel_size_cnn"]),
            n_ffn_layers_cnn=model_setup["n_ffn_layers_cnn"],
        )

    elif name == "CvT3_CvT3":
        return CvT3_CvT3(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list_1=tuple(model_setup["n_CP_blocks_1"]),
            CTemb_channels_list_1=tuple(model_setup["CTemb_channels_1"]),
            CP_channels_list_1=tuple(model_setup["CP_channels_1"]),
            attn_heads_list_1=tuple(model_setup["attn_heads_1"]),
            kernel_1=tuple(model_setup["kernel_1"]),
            final_architecture_1=ast.literal_eval(model_setup["final_architecture_1"]),
            n_CP_blocks_list_2=tuple(model_setup["n_CP_blocks_2"]),
            CTemb_channels_list_2=tuple(model_setup["CTemb_channels_2"]),
            CP_channels_list_2=tuple(model_setup["CP_channels_2"]),
            attn_heads_list_2=tuple(model_setup["attn_heads_2"]),
            kernel_2=tuple(model_setup["kernel_2"]),
            final_architecture_2=ast.literal_eval(model_setup["final_architecture_2"]),
            phasors=model_setup["phasors"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )

    elif name == "CvT3_CNNPh":
        return CvT3_CNNPh(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            cnnph_channels=model_setup["cnnph_channels"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "CvT3_EDPPh":
        return CvT3_EDPPh(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            edpph_channels=model_setup["edpph_channels"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "CvT3_CNNClsf":
        return CvT3_CNNClsf(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list=tuple(model_setup["n_CP_blocks"]),
            CTemb_channels_list=tuple(model_setup["CTemb_channels"]),
            CP_channels_list=tuple(model_setup["CP_channels"]),
            attn_heads_list=tuple(model_setup["attn_heads"]),
            kernel=tuple(model_setup["kernel"]),
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            cnnclsf_channels=tuple(model_setup["cnnclsf_channels"]),
            n_classes=model_setup["n_classes"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )

    elif name == "CvTaps_CvTaps":
        return CvTaps_CvTaps(
            lattice_size=tuple(model_setup["lattice_size"]),
            n_CP_blocks_list_1=tuple(model_setup["n_CP_blocks_1"]),
            channels_list_1=tuple(model_setup["channels_1"]),
            attn_heads_list_1=tuple(model_setup["attn_heads_1"]),
            strides_list_1=tuple([tuple(st) for st in model_setup["strides_1"]]),
            kernel_1=tuple(model_setup["kernel_1"]),
            final_architecture_1=ast.literal_eval(model_setup["final_architecture_1"]),
            n_CP_blocks_list_2=tuple(model_setup["n_CP_blocks_2"]),
            channels_list_2=tuple(model_setup["channels_2"]),
            attn_heads_list_2=tuple(model_setup["attn_heads_2"]),
            strides_list_2=tuple([tuple(st) for st in model_setup["strides_2"]]),
            kernel_2=tuple(model_setup["kernel_2"]),
            final_architecture_2=ast.literal_eval(model_setup["final_architecture_2"]),
            phasors=model_setup["phasors"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )
    elif name == "ViT2D_CNNClsf":
        return ViT2D_CNNClsf(
            lattice_size=tuple(model_setup["lattice_size"]),
            token_size=tuple(model_setup["token_size"]),
            embedding_d=model_setup["embedding_d"],
            n_heads=model_setup["n_heads"],
            n_blocks=model_setup["n_blocks"],
            n_ffn_layers=model_setup["n_ffn_layers"],
            final_architecture=ast.literal_eval(model_setup["final_architecture"]),
            cnnclsf_channels=tuple(model_setup["cnnclsf_channels"]),
            n_classes=model_setup["n_classes"],
            symm_2D=model_setup["symm_2D"],
            symm_Z2=model_setup["symm_Z2"],
            trivial_Z2=model_setup["trivial_Z2"],
        )


