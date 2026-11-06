from itertools import product

from .model import REGISTRED_MODELS


class ModelFactory:
    """
    Clase para inicializar y crear modelos con diferentes configuraciones.
    """

    def __init__(self, size, setup):

        self.size = size
        self.iter_mode = setup["mode"]
        self.kwargs_lattice = setup["kwargs_lattice"]
        self.S_operators = setup["S_operators"]

        self.name = setup["CM"]["selection"]
        self.setup = setup["CM"][self.name]

        self.params = {}

    @classmethod
    def init(cls, setup):
        """
        Initializes models from a given setup dictionary, following
        the `get_setup` structure. Tipically used for loading saved configurations.
        Returns the model instance.
        """
        self = cls.__new__(cls)

        self.size = setup["size"]
        self.name = setup["name"]
        self.params = setup["params"]
        self.kwargs_lattice = setup["kwargs_lattice"]
        self.S_operators = setup["S_operators"]

        return self

    def get_iterator(self):
        """
        Crea un iterador sobre los parámetros de setup.

        mode = "for" → todas las combinaciones (producto cartesiano)
        mode = "zip" → elementos emparejados (tipo zip)
        """
        # Split params ended with "_list"
        list_params = {k[:-5]: v for k, v in self.setup.items() if k.endswith("_list")}
        other_params = {k: v for k, v in self.setup.items() if not k.endswith("_list")}

        # Iterator based on mode
        if self.iter_mode == "for":
            keys = list(list_params.keys())
            for combo in product(*list_params.values()):
                yield {**dict(zip(keys, combo)), **other_params}

        elif self.iter_mode == "zip":
            keys = list(list_params.keys())
            for combo in zip(*list_params.values()):
                yield {**dict(zip(keys, combo)), **other_params}

        else:
            raise ValueError("El modo debe ser 'for' o 'zip'.")

    def preprocess_params(self, name, params):
        """
        Preprocesa los parámetros si es necesario.
        """

        if name == "Oxalate":
            keys = [["strength", "theta", "phi"], ["J", "K", "Gamma"]]
            if all(key in params.keys() for key in keys[0]):
                st = params.pop("strength")
                th = params.pop("theta")
                ph = params.pop("phi")
                params["couplings"] = [st, th, ph]

            elif all(key in params.keys() for key in keys[1]):
                J = params.pop("J")
                K = params.pop("K")
                G = params.pop("Gamma")
                params["couplings"] = [J, K, G]
            else:
                assert "couplings" in params, ValueError(
                    "If there is no explicit name for coupling model parameters, "
                    "Oxalate params must have a 'couplings' key of length = 3"
                )
                assert (
                    len(params["couplings"]) == 3
                ), "Oxalate params must have a 'couplings' key of length = 3"

        return params

    def get_model(self):
        """
        Crea una instancia del modelo con los parámetros dados.
        """

        ModelClass = REGISTRED_MODELS.get(self.name)
        if ModelClass is None:
            raise ValueError(f"Modelo '{self.name}' no registrado.")

        params = self.preprocess_params(self.name, self.params)

        return ModelClass(self.size, **params, **self.kwargs_lattice)

    def get_params(self):
        for params in self.get_iterator():
            self.params = params
            yield params

    def get_setup(self):
        setup = {
            "size": self.size,
            "name": self.name,
            "params": self.params,
            "kwargs_lattice": self.kwargs_lattice,
            "S_operators": self.S_operators,
        }
        return setup
