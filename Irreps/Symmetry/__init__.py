from .Simmetries import Z2, Traslation

SIMMETRY = {
    'Z2':Z2,
    'Traslation': Traslation
}

# EXTERNAL ARGS

LATTICE_SIZE = {
    'lattice_size':[
        'Traslation'
    ]
}

EXTERNAL_ARGS = {
    **LATTICE_SIZE
}