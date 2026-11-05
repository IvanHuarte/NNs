"""
Diagram of an arbitrary Hydra/NeuralNetwork architecture.

    Wrappers: NNWrapper → Z2 [ψ(x) − ψ(−x)]
    Root: SplitTraining
                 X_in
                   ║
         ╔═════════╩══════════╗
         ║                    ║
     <Modulus>             <Phase>
         ║                    ║
         ╔════╩════╗ <----- Transversal (0)
    ╔════╩═════╗ ...
    ║ CNN (00) ║ ...
         ...
         ╚════╦════╝
              Σ
              ║

- Boxes are single modules, with their path code (the codes used in load
  instructions and schedule modes).
- Flux modules are labelled "<----- Name (code)" at the height of the node
  where their branches split (or where their chain starts).
- The symbol just above the node where branches meet is the merge operation
  (legend at the end).
- Top wrappers (NNWrapper/ARNNWrapper and symmetry wrappers) are listed in the
  header; if the network starts with a flux module it is given as "Root" and
  not labelled in the drawing.

    from NN_module.NN.arch_diagram import draw_architecture
    print(draw_architecture(hydra.setup, params))
"""

from dataclasses import dataclass

from NN_module.NN import (
    __all_flux__,
    __all_symm__,
    __all_wrappers__,
    submodule_tags,
    symm_submodule_dict,
    wrapper_submodule_dict,
)
from NN_module.NN.utils import get_code2path_flatten

GAP = 3  # columns between parallel branches
ARROW = "<----- "


# ----------------------------------------------------------------- blocks


@dataclass
class Block:
    """Lines with a vertical axis (column) where the flow enters
    at the top and leaves at the bottom."""

    lines: list
    axis: int

    @property
    def width(self):
        return max((len(l) for l in self.lines), default=0)

    @property
    def height(self):
        return len(self.lines)


def _pad(block, left, width):
    return Block([(" " * left + l).ljust(width) for l in block.lines], block.axis + left)


def box(text):
    """Single module box."""
    w = len(text) + 4
    c = w // 2
    top = "╔" + "═" * (w - 2) + "╗"
    bot = "╚" + "═" * (w - 2) + "╝"
    return Block(
        [top[:c] + "╩" + top[c + 1 :], "║ " + text + " ║", bot[:c] + "╦" + bot[c + 1 :]],
        c,
    )


def text(t):
    """Free text on the flow (e.g. X_in, branch names)."""
    return Block([t], len(t) // 2)


def connector(label=None):
    """Vertical connector with a flux-module label on its right (None if
    there is no label: nothing to draw)."""
    if not label:
        return None
    return Block(["║  " + label], 0)


def add_label(block, row, label):
    """Append a flux-module label to the right of line `row`."""
    if not label:
        return block
    lines = list(block.lines)
    lines[row] = lines[row].rstrip() + " " + label
    return Block(lines, block.axis)


def stack(blocks, connect=True):
    """Blocks one below the other, aligned on their axis."""
    blocks = [b for b in blocks if b is not None and b.height]
    axis = max(b.axis for b in blocks)
    width = max(axis - b.axis + b.width for b in blocks)
    lines = []
    for i, b in enumerate(blocks):
        if i and connect:
            lines.append((" " * axis + "║").ljust(width))
        lines.extend(_pad(b, axis - b.axis, width).lines)
    return Block(lines, axis)


def parallel(branches, label=None, merge=None):
    """
    Branches side by side, split from a common input and merged at a node.
    `label` (flux module) goes at the right of the split node; `merge`
    (operation symbol) just above the merge node.

       ╔════╩════╗ <----- label
       ║         ║
     branch    branch
       ║         ║
       ║    Σ    ║
       ╚════╦════╝
    """
    if len(branches) == 1:
        return stack(branches)

    positions, x = [], 0
    for b in branches:
        positions.append(x)
        x += b.width + GAP
    width = x - GAP
    axes = [p + b.axis for p, b in zip(positions, branches)]
    axis = (axes[0] + axes[-1]) // 2
    height = max(b.height for b in branches)

    def fits(start, n):
        span = set(range(start, start + n))
        return start >= 0 and start + n <= width and not span & set(axes)

    def bar(first, last, inner, center):
        row = [" "] * width
        for i in range(axes[0], axes[-1] + 1):
            row[i] = "═"
        for a in axes[1:-1]:
            row[a] = inner
        row[axes[0]], row[axes[-1]] = first, last
        row[axis] = "╬" if axis in axes[1:-1] else center
        return row

    def verticals():
        row = [" "] * width
        for a in axes:
            row[a] = "║"
        return row

    rows = [bar("╔", "╗", "╦", "╩"), verticals()]
    padded = []
    for p, b in zip(positions, branches):
        filler = (" " * b.axis + "║").ljust(b.width)
        lines = [l.ljust(b.width) for l in b.lines] + [filler] * (height - b.height)
        padded.append((p, lines))
    for i in range(height):
        row = [" "] * width
        for p, lines in padded:
            for j, ch in enumerate(lines[i]):
                if ch != " ":
                    row[p + j] = ch
        rows.append(row)

    # Merge symbol just above the merge node (if it fits between branches)
    merge_row = verticals()
    merge_bar = bar("╚", "╝", "╩", "╦")
    suffix = ""
    if merge:
        n = len(merge)
        # Centered above the merge node; if a branch runs through it, just
        # to its right; otherwise after the merge node.
        for start in (axis - n // 2, axis + 2):
            if fits(start, n):
                for i, ch in enumerate(merge):
                    merge_row[start + i] = ch
                break
        else:
            suffix = " " + merge
    rows += [merge_row, merge_bar]

    lines = ["".join(r) for r in rows]
    lines[-1] = lines[-1].rstrip() + suffix
    return add_label(Block(lines, axis), 0, label)


# ------------------------------------------------------- module semantics

# symbol: meaning (merge operations)
TRANSVERSAL_OPS = {
    "sum": ("Σ", "sum of the branches"),
    "sum_angles": ("Σ mod 2π", "sum of the branches wrapped to (−π, π]"),
    "product": ("Π", "product of the branches"),
    "mean": ("⟨·⟩", "mean of the branches"),
    "modulus_marshall": ("log(b₁·e^b₀)", "b₀ modulus, b₁ sign"),
    "None": ("[ ]", "branches stacked, no merge"),
}
SPLIT_OP = ("+i", "Modulus + i·Phase  (log|ψ| + i·φ)")
SEQV_OP = ("+iφ_k", "log ψ + i·φ_k,  φ_k = arg Σ_r χ_k(r)·f(r)")
SINGLE_MODES = {"modulus": "x + 0i", "phase": "1 + i·x"}


def _children(setup):
    return {
        k: v
        for k, v in setup.items()
        if isinstance(v, dict) and any(tag in k for tag in submodule_tags)
    }


def _attrs(setup):
    return {k: v for k, v in setup.items() if not isinstance(v, dict)}


def _symm_op(module, attrs):
    if module == "Z2":
        if attrs.get("symm_modulus", False):
            return "log|ψ| = mean(log|ψ(x)|, log|ψ(−x)|)"
        return "ψ(x) + ψ(−x)" if attrs.get("trivial_Z2", True) else "ψ(x) − ψ(−x)"
    irrep = tuple(attrs.get("irrep", (0, 0)))
    op = f"Σ_T χ_k(T)·ψ(T x), k={irrep}"
    return op + ", anchor" if attrs.get("use_anchor", False) else op


class _Drawer:
    def __init__(self, path2code, show_attrs):
        self.path2code = path2code
        self.show_attrs = show_attrs
        self.legend = {}

    def code(self, path):
        code = self.path2code.get(tuple(path))
        return f" ({code})" if code is not None else ""

    def flux_label(self, module, path, extra="", is_root=False):
        if is_root:
            return None
        return ARROW + module + self.code(path) + (f" [{extra}]" if extra else "")

    def merge(self, op):
        symbol, meaning = op
        self.legend[symbol] = meaning
        return symbol

    def draw(self, node, path, is_root=False):
        module = node["module"]
        setup = node["setup"]
        children = _children(setup)
        attrs = _attrs(setup)

        # Symmetry wrapper inside the network (top ones go in the header)
        if module in __all_symm__ and "Wrap" in setup:
            inner = self.draw(setup["Wrap"], path + [symm_submodule_dict[module]])
            label = self.flux_label(module, path, _symm_op(module, attrs))
            return stack([connector(label), inner], connect=False)

        if module == "Sequential":
            blocks = [self.draw(v, path + [k]) for k, v in children.items()]
            label = self.flux_label(module, path, is_root=is_root)
            return stack([connector(label), stack(blocks)], connect=False)

        if module == "SingleModule":
            (k, v), = children.items()
            mode = attrs.get("mode")
            label = self.flux_label(module, path, SINGLE_MODES.get(mode, ""), is_root)
            return stack([connector(label), self.draw(v, path + [k])], connect=False)

        if module in ("SplitTraining", "Transversal"):
            branches = [self.draw(v, path + [k]) for k, v in children.items()]
            if module == "SplitTraining":
                branches = [
                    stack([text(f"<{k}>"), b]) for k, b in zip(children, branches)
                ]
                merge = self.merge(SPLIT_OP)
            else:
                operation = str(attrs.get("operation", "sum"))
                op = TRANSVERSAL_OPS.get(operation, (operation, operation))
                if attrs.get("post_norm", False):
                    op = ("LN→" + op[0], "LayerNorm, then " + op[1])
                merge = self.merge(op)
            label = self.flux_label(module, path, is_root=is_root)
            return parallel(branches, label=label, merge=merge)

        if module in ("Sequivariant", "SequivariantX"):
            irrep = tuple(attrs.get("irrep", (0, 0)))
            blocks = [self.draw(v, path + [k]) for k, v in children.items()]
            label = self.flux_label(module, path, f"k={irrep}", is_root)
            phase = text("φ_k")
            merge = self.merge(SEQV_OP)
            if module == "Sequivariant":
                # phase from the features entering the last module (head)
                fork = parallel([blocks[-1], phase], label=label, merge=merge)
                return stack(blocks[:-1] + [fork])
            # SequivariantX: phase from the input configuration
            return parallel([stack(blocks), phase], label=label, merge=merge)

        # Single module: the only boxes
        name = module + self.code(path)
        if self.show_attrs and attrs:
            name += " {" + ", ".join(f"{k}={v}" for k, v in attrs.items()) + "}"
        return box(name)


def draw_architecture(setup, params=None, show_attrs=False):
    """
    Text diagram of a network setup (Hydra.setup / NeuralNetwork.setup).

    params: parameters of the model (with or without the 'params' level).
        If given, every module with parameters shows its path code.
    show_attrs: also print the hyperparameters of the single modules.
    """
    path2code = {}
    if params is not None:
        params = params.get("params", params)
        path2code = {tuple(v): k for k, v in get_code2path_flatten(params).items()}
    drawer = _Drawer(path2code, show_attrs)

    # Header: chain of top wrappers
    wrappers, node, path = [], setup, []
    if node["module"] in __all_wrappers__:
        wrappers.append(node["module"])
        path = [wrapper_submodule_dict[node["module"]]]
        node = {"module": node["setup"]["module"], "setup": node["setup"]["setup"]}
    while node["module"] in __all_symm__ and "Wrap" in node["setup"]:
        attrs = _attrs(node["setup"])
        wrappers.append(f"{node['module']} [{_symm_op(node['module'], attrs)}]")
        path = path + [symm_submodule_dict[node["module"]]]
        node = node["setup"]["Wrap"]

    is_root = node["module"] in __all_flux__
    header = []
    if wrappers:
        header.append("Wrappers: " + " → ".join(wrappers))
    if is_root:
        root = node["module"]
        attrs = _attrs(node["setup"])
        if root in ("Sequivariant", "SequivariantX"):
            root += f" [k={tuple(attrs.get('irrep', (0, 0)))}]"
        elif root == "SingleModule" and attrs.get("mode") in SINGLE_MODES:
            root += f" [{SINGLE_MODES[attrs['mode']]}]"
        header.append(f"Root: {root}")

    autoregressive = setup["module"] == "ARNNWrapper"
    body = drawer.draw(node, path, is_root=is_root)
    diagram = stack(
        [
            text("X_in"),
            body,
            text("log ψ(x)" + ("  [autoregressive]" if autoregressive else "")),
        ]
    )

    lines = header + ([""] if header else []) + [l.rstrip() for l in diagram.lines]
    if drawer.legend:
        lines.append("")
        lines += [f"{sym:>6} : {meaning}" for sym, meaning in drawer.legend.items()]
    return "\n".join(lines)
