from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.data.wavefactor import WaveFactor

from numpy import ndarray
from typing import List, Tuple


class WaveCalculator(AlphaCalculator):
    def __init__(self, **kwargs):
        self.wavefactor = WaveFactor(**kwargs)

    def _calc_alpha(self, expr: Expression) -> ndarray:
        return self.wavefactor.factor(str(expr))

    def _calc_IC(self, value1: ndarray, value2: ndarray) -> float:
        # return batch_pearsonr(value1, value2).mean().item()
        return 0.0

    def _calc_rIC(self, value1: ndarray, value2: ndarray) -> float:
        return 0.0

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> ndarray:
        n = len(exprs)
        factors: List[ndarray] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        ensemble_value = self.make_ensemble_alpha(exprs, weights)
        return self._calc_IC(ensemble_value, self.target_value)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        ensemble_value = self.make_ensemble_alpha(exprs, weights)
        return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        if len(exprs) <= 0:
            return (0.0, 0.0)
        ensemble_value = self.make_ensemble_alpha(exprs, weights)
        return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)