from ensemble.base import BaseEnsemble, EnsembleType
from ensemble.independent import IndependentEnsemble


class EnsembleFactory:
    def __new__(cls, model_generator_function, models_n, ensemble_type: str) -> BaseEnsemble:
        if ensemble_type == EnsembleType.INDEPENDENT.value:
            return IndependentEnsemble(model_generator_function, models_n)
