"""
Architecture diagram (Hydra/NN/arch_diagram.py).

Run (CPU is enough):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest pytests/test_diagram -q
"""

import copy
import os
import sys

import netket as nk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Hydra.NN.Hydra import Hydra

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
        "Wrappers: NNWrapper", "Root: SplitTraining",
        "<----- Transversal (0)", "║ MLP (00) ║", "║ MLP (01) ║",
        "<----- Sequential (1)", "║ ViT2D (10) ║", "║ OutputHead (11) ║",
        "<Modulus>", "<Phase>", "X_in  (B, N) → (B, Lx, Ly, 1)", "log ψ(x)",
        "Σ : sum of the branches", "+i : Modulus + i·Phase",
    ]:
        assert piece in text, piece
    # Only single modules are boxed; the root flux module is not drawn
    assert "║ Transversal" not in text and "║ Sequential" not in text
    assert "<----- SplitTraining" not in text


def test_modules_without_parameters_have_no_code():
    text = hydra(
        {"SplitTraining": {"Sequential": {"ViT2D": {}, "OutputHead": {}}, "MarshallSign": {}}}
    ).draw_architecture()
    assert "║ MarshallSign ║" in text


def test_symmetries_show_their_operation():
    text = hydra(
        {"Sequivariant": {"ViT2D": {}, "OutputHead": {}, "irrep": [1, 0]}}, z2=False
    ).draw_architecture()
    assert "Wrappers: NNWrapper → Z2 [ψ(x) − ψ(−x)]" in text
    assert "Root: Sequivariant [k=(1, 0)]" in text
    assert "+iφ_k" in text and "φ_k = arg Σ_r χ_k(r)·f(r)" in text


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


def test_merge_symbol_is_just_above_the_merge_node():
    text = hydra(
        {"Sequential": {"ViT2D": {}, "Transversal": {
            "OutputHead": {}, "OutputHead_1": {}, "operation": "product"}}}
    ).draw_architecture()
    lines = text.splitlines()
    i = next(i for i, l in enumerate(lines) if "Π" in l and ":" not in l)
    merge_node = lines[i + 1].index("╦")
    assert lines[i + 1].lstrip().startswith("╚")
    assert abs(lines[i].index("Π") - merge_node) <= 1
    assert "Root: Sequential" in text and "<----- Transversal (1)" in text
