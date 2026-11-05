"""
Architecture diagram (NN_module/NN/arch_diagram.py).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_diagram -q
"""

import copy
import os
import sys

import netket as nk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from NN_module.NN.Hydra import Hydra

LATTICE = (4, 4)
STORAGE = {
    "ViT2D": {
        "token_size": [1, 2], "embedding_d": 8, "n_heads": 2,
        "n_blocks": 1, "n_ffn_layers": 1, "final_architecture": None,
    },
    "OutputHead": {"d_model": 8, "only_phase": False},
    "MLP": {"hidden_alpha": [1], "activation": ["swish"], "final_architecture": [1], "only_phase": False},
    "MarshallSign": {},
}
SYMM = {
    "Traslation": {"on": False, "irrep": [0, 0], "use_anchor": False, "save_memory": False},
    "Z2": {"on": False, "trivial_Z2": True, "symm_modulus": False},
}


def hydra(template, z2=None):
    symm = copy.deepcopy(SYMM)
    if z2 is not None:
        symm["Z2"].update(on=True, trivial_Z2=z2)
    cfg = {
        "print_arch": False,
        "load_model": {"parameters": False, "sampler": False},
        "storage": copy.deepcopy(STORAGE),
        "symm_wrapper": symm,
        "selection": "T",
        "T": {"stage0": {"template": template}},
    }
    return Hydra(cfg, lattice_size=LATTICE, hilbert=nk.hilbert.Spin(0.5, 16))


def test_split_training_with_transversal():
    text = hydra(
        {"SplitTraining": {
            "Transversal": {"MLP": {}, "MLP_1": {}},
            "Sequential": {"ViT2D": {}, "OutputHead": {}},
        }}
    ).draw_architecture()
    for piece in [
        "Transversal (0)", "MLP (00)", "MLP (01)",
        "Sequential (1)", "ViT2D (10)", "OutputHead (11)",
        "<Modulus>", "<Phase>", "Σ branches", "Modulus + i·Phase",
        "X_in", "log ψ(x)",
    ]:
        assert piece in text, piece


def test_modules_without_parameters_have_no_code():
    text = hydra(
        {"SplitTraining": {"Sequential": {"ViT2D": {}, "OutputHead": {}}, "MarshallSign": {}}}
    ).draw_architecture()
    assert "║ MarshallSign ║" in text


def test_symmetries_show_their_operation():
    text = hydra(
        {"Sequivariant": {"ViT2D": {}, "OutputHead": {}, "irrep": [1, 0]}}, z2=False
    ).draw_architecture()
    assert "Z2 [symmetrization]" in text
    assert "ψ(x) − ψ(−x)" in text
    assert "irrep k=(1, 0)" in text
    assert "φ_k = arg Σ_r χ_k(r)·f(r)" in text
    assert "log ψ + i·φ_k" in text


def test_connectors_are_continuous():
    """Every vertical connector in a line continues in the next line."""
    text = hydra(
        {"SplitTraining": {
            "Transversal": {"MLP": {}, "MLP_1": {}},
            "Sequential": {"ViT2D": {}, "OutputHead": {}},
        }}
    ).draw_architecture()
    lines = text.splitlines()
    width = max(map(len, lines))
    lines = [l.ljust(width) for l in lines]
    box_chars = set("═║╔╗╚╝╦╩╬╭╮╰╯─│╨╥")
    comes_from_above = set("║╩╝╚╨╬╔╗")
    for a, b in zip(lines, lines[1:]):
        for i, ch in enumerate(a):
            if ch in "║╦╥╬":
                # below: a connector, or a text label (branch name, X_out)
                assert b[i] in comes_from_above or (
                    b[i] != " " and b[i] not in box_chars
                ), (a, b, i)


def test_users_reference_architecture_with_cnn(capsys):
    """SplitTraining(Transversal(CNN, MLP), CvT)-like net: CNN gets
    lattice_size injected and the diagram is printed by update_info."""
    storage_cnn = {
        "channels": [4], "strides": [[1, 1]], "kernel": [3, 3], "use_pooling": False,
        "pooling_strides": [[1, 1]], "final_architecture": [1],
    }
    STORAGE["CNN"] = storage_cnn
    try:
        h = hydra({"SplitTraining": {
            "Transversal": {"CNN": {}, "MLP": {}},
            "Sequential": {"ViT2D": {}, "OutputHead": {}},
        }})
    finally:
        STORAGE.pop("CNN")
    out = capsys.readouterr().out
    assert "ARCHITECTURE INFO" in out and "CNN (00)" in out and "MLP (01)" in out
