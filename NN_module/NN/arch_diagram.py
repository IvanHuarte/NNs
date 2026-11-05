"""
Box diagram of an arbitrary Hydra/NeuralNetwork architecture.

Each module is drawn as a box with its name and path code (the same codes
used in load instructions and in the schedule modes). Modules with branches
(SplitTraining, Transversal, ...) open the branches in parallel and close
them in an operation box (single line) that shows how they are merged.

    from NN_module.NN.arch_diagram import draw_architecture
    print(draw_architecture(hydra.setup, params))

Layout: every block is a list of equal-width lines with a vertical axis
(column) where the flow enters at the top and leaves at the bottom.
"""

from dataclasses import dataclass

from NN_module.NN import (
    __all_symm__,
    __all_wrappers__,
    submodule_tags,
    symm_submodule_dict,
    wrapper_submodule_dict,
)
from NN_module.NN.utils import get_code2path_flatten

GAP = 3  # columns between parallel branches


# ----------------------------------------------------------------- blocks


@dataclass
class Block:
    lines: list
    axis: int

    @property
    def width(self):
        return max((len(l) for l in self.lines), default=0)

    @property
    def height(self):
        return len(self.lines)


def _pad(block, left, width):
    lines = [" " * left + l for l in block.lines]
    return Block([l.ljust(width) for l in lines], block.axis + left)


def _connector(axis, width, char="║"):
    return (" " * axis + char).ljust(width)


def box(text, double=True):
    """Module box (double line) or operation box (single line)."""
    w = len(text) + 4
    c = w // 2
    if double:
        top = "╔" + "═" * (w - 2) + "╗"
        mid = "║ " + text + " ║"
        bot = "╚" + "═" * (w - 2) + "╝"
        top = top[:c] + "╩" + top[c + 1 :]
        bot = bot[:c] + "╦" + bot[c + 1 :]
    else:
        top = "╭" + "─" * (w - 2) + "╮"
        mid = "│ " + text + " │"
        bot = "╰" + "─" * (w - 2) + "╯"
        top = top[:c] + "╨" + top[c + 1 :]
        bot = bot[:c] + "╥" + bot[c + 1 :]
    return Block([top, mid, bot], c)


def label(text):
    """Free text on the axis (e.g. X_in)."""
    return Block([text], len(text) // 2)


def stack(blocks, connect=True):
    """Blocks one below the other, aligned on their axis."""
    blocks = [b for b in blocks if b is not None and b.height]
    axis = max(b.axis for b in blocks)
    width = max(axis - b.axis + b.width for b in blocks)
    lines = []
    for i, b in enumerate(blocks):
        if i and connect:
            lines.append(_connector(axis, width))
        lines.extend(_pad(b, axis - b.axis, width).lines)
    return Block(lines, axis)


def parallel(branches, header=None, op=None):
    """
    Branches side by side, opened from a common input and merged with `op`.

          header
            ║
       ╔════╩════╗
     branch    branch
       ╚════╦════╝
        ╭───╨───╮
        │  op   │
        ╰───╥───╯
    """
    if len(branches) == 1:
        return stack([b for b in [header, branches[0], op] if b is not None])

    # Place branches side by side
    positions, x = [], 0
    for b in branches:
        positions.append(x)
        x += b.width + GAP
    width = x - GAP
    axes = [p + b.axis for p, b in zip(positions, branches)]
    axis = (axes[0] + axes[-1]) // 2
    height = max(b.height for b in branches)

    def bar(first, last, inner, center):
        row = [" "] * width
        for i in range(axes[0], axes[-1] + 1):
            row[i] = "═"
        for a in axes[1:-1]:
            row[a] = inner
        row[axes[0]], row[axes[-1]] = first, last
        row[axis] = "╬" if axis in axes[1:-1] else center
        return "".join(row)

    def verticals():
        row = [" "] * width
        for a in axes:
            row[a] = "║"
        return "".join(row)

    rows = [bar("╔", "╗", "╦", "╩"), verticals()]
    padded = []
    for p, b in zip(positions, branches):
        lines = list(b.lines) + [_connector(b.axis, b.width)] * (height - b.height)
        padded.append((p, [l.ljust(b.width) for l in lines]))
    for i in range(height):
        row = [" "] * width
        for p, lines in padded:
            for j, ch in enumerate(lines[i]):
                if ch != " ":
                    row[p + j] = ch
        rows.append("".join(row))
    rows += [verticals(), bar("╚", "╝", "╩", "╦")]

    body = Block(rows, axis)
    parts = [header, body, op] if header is not None else [body, op]
    return stack([p for p in parts if p is not None])


# ------------------------------------------------------- module semantics

# How the outputs of the branches are combined (op box text)
TRANSVERSAL_OPS = {
    "sum": "Σ branches",
    "sum_angles": "Σ branches mod 2π",
    "product": "Π branches",
    "mean": "mean of branches",
    "modulus_marshall": "log(b1·exp(b0))",
    "None": "stack (no merge)",
}


def _children(setup):
    return {
        k: v
        for k, v in setup.items()
        if isinstance(v, dict) and any(tag in k for tag in submodule_tags)
    }


def _attrs(setup):
    return {k: v for k, v in setup.items() if not isinstance(v, dict)}


class _Drawer:
    def __init__(self, path2code, show_attrs):
        self.path2code = path2code
        self.show_attrs = show_attrs

    def name(self, module, path, attrs=None, extra=""):
        code = self.path2code.get(tuple(path))
        text = module + (f" ({code})" if code is not None else "")
        if extra:
            text += f" [{extra}]"
        if self.show_attrs and attrs:
            text += " {" + ", ".join(f"{k}={v}" for k, v in attrs.items()) + "}"
        return text

    def draw(self, node, path):
        module = node["module"]
        setup = node["setup"]
        children = _children(setup)
        attrs = _attrs(setup)

        # --- symmetry wrappers: evaluate the inner net on transformed inputs
        if module in __all_symm__ and "Wrap" in setup:
            inner = self.draw(setup["Wrap"], path + [symm_submodule_dict[module]])
            if module == "Z2":
                if attrs.get("symm_modulus", False):
                    op = "log|ψ| ← mean(log|ψ(x)|, log|ψ(−x)|)"
                elif attrs.get("trivial_Z2", True):
                    op = "ψ(x) + ψ(−x)"
                else:
                    op = "ψ(x) − ψ(−x)"
            else:
                irrep = attrs.get("irrep", (0, 0))
                op = f"Σ_T χ_k(T)·ψ(T x)  k={tuple(irrep)}"
                if attrs.get("use_anchor", False):
                    op += "  (anchor)"
            header = box(self.name(module, path, extra="symmetrization"))
            return stack([header, inner, box(op, double=False)])

        # --- flux modules
        if module == "Sequential":
            blocks = [self.draw(v, path + [k]) for k, v in children.items()]
            return stack([box(self.name(module, path))] + blocks)

        if module == "SingleModule":
            (k, v), = children.items()
            mode = attrs.get("mode")
            ops = {"modulus": "x + 0i", "phase": "1 + i·x"}
            parts = [box(self.name(module, path)), self.draw(v, path + [k])]
            if mode in ops:
                parts.append(box(ops[mode], double=False))
            return stack(parts)

        if module in ("SplitTraining", "Transversal"):
            branches = [self.draw(v, path + [k]) for k, v in children.items()]
            if module == "SplitTraining":
                # Name the role of each branch
                branches = [
                    stack([label(f"<{k}>"), b], connect=False)
                    for k, b in zip(children, branches)
                ]
                op = "log|ψ| + i·φ   (Modulus + i·Phase)"
            else:
                operation = attrs.get("operation", "sum")
                op = TRANSVERSAL_OPS.get(str(operation), str(operation))
                if attrs.get("post_norm", False):
                    op = "LayerNorm → " + op
            return parallel(
                branches,
                header=box(self.name(module, path)),
                op=box(op, double=False),
            )

        if module in ("Sequivariant", "SequivariantX"):
            irrep = tuple(attrs.get("irrep", (0, 0)))
            blocks = [self.draw(v, path + [k]) for k, v in children.items()]
            header = box(self.name(module, path, extra=f"irrep k={irrep}"))
            phase = box("φ_k = arg Σ_r χ_k(r)·f(r)", double=False)
            merge = box("log ψ + i·φ_k", double=False)
            if module == "Sequivariant":
                # phase from the features entering the last module (head)
                body = stack(blocks[:-1]) if len(blocks) > 1 else None
                fork = parallel([blocks[-1], phase], op=merge)
                return stack([b for b in [header, body, fork] if b is not None])
            # SequivariantX: phase from the input configuration
            return stack([header, parallel([stack(blocks), phase], op=merge)])

        # --- single (leaf) module
        return box(self.name(module, path, attrs))


def draw_architecture(setup, params=None, show_attrs=False):
    """
    Text diagram of a network setup (Hydra.setup / NeuralNetwork.setup).

    params: parameters of the model (dict with or without the 'params'
        level). If given, every module with parameters shows its path code.
    show_attrs: also print the hyperparameters of the leaf modules.
    """
    path2code = {}
    if params is not None:
        params = params.get("params", params)
        path2code = {tuple(v): k for k, v in get_code2path_flatten(params).items()}

    drawer = _Drawer(path2code, show_attrs)

    module = setup["module"]
    if module in __all_wrappers__:
        inner_path = [wrapper_submodule_dict[module]]
        inner = setup["setup"]
        body = drawer.draw({"module": inner["module"], "setup": inner["setup"]}, inner_path)
        title = "X_in  (B, N)" if module == "ARNNWrapper" else "X_in  (B, N) → (B, Lx, Ly, 1)"
        out = "log ψ(x)" + ("   [autoregressive]" if module == "ARNNWrapper" else "")
        diagram = stack([label(title), body, label(out)])
    else:
        diagram = drawer.draw(setup, [])

    return "\n".join(l.rstrip() for l in diagram.lines)
