from .Simmetries import Z2, Traslation, Cn, Refl_H, Refl_V, Refl_D, Refl_AD

SIMMETRY = {
    'Z2':Z2,
    'Traslation': Traslation,
    "Cn": Cn,
    "Refl_H": Refl_H,
    "Refl_V": Refl_V,
    "Refl_D": Refl_D,
    "Refl_AD": Refl_AD
}

# EXTERNAL ARGS

LATTICE_SIZE = {
    'lattice_size':[
        'Traslation',
        'Cn',
        "Refl_H",
        "Refl_V",
        "Refl_D",
        "Refl_AD"
    ]
}

EXTERNAL_ARGS = {
    **LATTICE_SIZE
}