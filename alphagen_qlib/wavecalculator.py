from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.data.wavecalc import wavecalc

from typing import List, Tuple
from legion import KTD
import pdb
import uuid

def genuid(prefix = ''):
    return prefix + uuid.uuid4().hex


class WaveCalculator(AlphaCalculator):
    def __init__(self, **kwargs):
        self.wavecalc = wavecalc(**kwargs)

    def _calc_alpha(self, expr: Expression) -> KTD:
        return self.wavecalc.factor([str(expr)])

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> KTD:
        pdb.set_trace()
        euid = genuid('alpgen.')
        wexprs = [f"{euid}.expr.{i} <- {str(e)}" for i, e in enumerate(exprs)]
        namemap = {n: i for i, n in enumerate(wexprs.keys())}

        facs = self.wavecalc.rawfactor(wexprs, burnin = 0) # keep the burns
        wf = KTD(facs[0].ks, facs[0].ts, facs[0].ds, 0.0)
        for n, v in facs:
            wf = wf + v * weights[namemap[n]]

        return {euid : wf}

    def calc_single_IC_ret(self, expr: Expression, custom_vs = None) -> float:
        pdb.set_trace()
        e = [str(expr)]
        ret = self.wavecalc.score(e, metrics='IC')
        return ret['exprs'][0]['ic.mean']

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self.calc_single_IC_ret(expr)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        ic = self.calc_single_IC_ret(expr)
        return ic, ic

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        pdb.set_trace()
        euid = genuid('alpgen.')
        wexprs = [f"{euid}.expr.{i} <- {str(e)}" for i, e in enumerate(exprs)]
        namemap = {n: i for i, n in enumerate(wexprs.keys())}

        facs = self.wavecalc.rawfactor(wexprs, burnin = 0) # keep the burns

        wf = KTD(facs[0].ks, facs[0].ts, facs[0].ds, 0.0)
        for n, v in facs:
            wf = wf + v * weights[namemap[n]]

        custom_vs = {euid : wf}
        return self.calc_singel_IC_ret(euid, custom_vs)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return self.calc_pool_IC_ret(exprs, weights)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        if len(exprs) <= 0:
            return (0.0, 0.0)
        ic = self.calc_pool_IC_ret(exprs, weights)
        return ic, ic
