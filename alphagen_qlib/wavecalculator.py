from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen_qlib.wavefactor import WaveFactor

from typing import List, Tuple
from legion import KTD
import pdb
import uuid
import re
import numpy as np

def genuid(prefix = ''):
    return prefix + uuid.uuid4().hex

class Qlib2Wave:
    ops = "$md_std_adjf $md_std_pcls $md_std_trdsts $md_std_cur $md_std_amt $md_std_lret $md_std_adjpcls $md_std_ret $md_std_low $md_std_cls $md_std_vol $md_std_hgh $md_std_opn $md_std_vwp".split(' ')
    lops = "md/std/Adjf md/std/Pcls md/std/Trdsts md/std/Cur md/std/Amt md/std/lRet md/std/AdjPcls md/std/Ret md/std/Low md/std/Cls md/std/Vol md/std/Hgh md/std/Opn md/std/Vwp".split(' ')

    origs = "Mul Sub Ref Mean Med Sum Std Var Max Min Mad Delta WMA EMA Cov Corr Constant".split(' ')
    repls = "Mult Minus TsDelay TsMean TsMedian TsSum TsStd TsVar TsMax TsMin TsMad TsDelta TsWma TsEma TsCov TsCorr Broadcast".split(' ')

    assert(len(origs) == len(repls))
    assert(len(ops) == len(lops))

    wdbtables = "AIndexValuation ASHAREBALANCESHEET ASHARECASHFLOW ASHARECONSENSUSROLLINGDATA_CAGR ASHARECONSENSUSROLLINGDATA_FTTM ASHARECONSENSUSROLLINGDATA_FY0 ASHARECONSENSUSROLLINGDATA_FY1 ASHARECONSENSUSROLLINGDATA_FY2 ASHARECONSENSUSROLLINGDATA_FY3 ASHARECONSENSUSROLLINGDATA_YOY ASHARECONSENSUSROLLINGDATA_YOY2 ASHAREEODDERIVATIVEINDICATOR ASHAREFINANCIALINDICATOR ASHAREINCOME ASHAREMARGINTRADE ASHAREMONEYFLOW AShareEnergyindex AShareEnergyindexADJ AShareHolderNumber AShareL2Indicators AShareTechIndicators AShareYield AShareswingReversetrend AShareswingReversetrendADJ Ashareintensitytrend AshareintensitytrendADJ"
    wdbmaps = {w.lower(): w for w in wdbtables.split(' ')}

    @staticmethod
    def handle_wdb(e):
        e = e.group(0)
        if not e.startswith('$wdb'):
            return e
        sep = e.split('_')
        # pdb.set_trace()
        return '/'.join(['wdb', sep[1], '_'.join(sep[2:]).upper()])

    @staticmethod
    def waveexpr(e):
        e = str(e)
        for o, r in zip(Qlib2Wave.origs, Qlib2Wave.repls):
            e = e.replace(o+"(", r+"(")

        for o, ml in zip(Qlib2Wave.ops, Qlib2Wave.lops):
            e = e.replace(o, ml)

        pattern = r"\$(.*?),"
        pattern = r"\$(.*?)(?=\W|$)"
        e = re.sub(pattern, Qlib2Wave.handle_wdb, e)

        e = f"Scale(Bound({e}))"

        return e


class WaveCalculator(AlphaCalculator):
    SCORER = 'ic.mean'

    def __init__(self, **kwargs):
        self.wavecalc = WaveFactor(**kwargs)

    def _calc_alpha(self, expr: Expression) -> KTD:
        return self.wavecalc.factor([Qlib2Wave.waveexpr(expr)])

    def _get_result(self, ret, field = None):
        field = field is None and self.SCORER or field
        if len(ret['exprs']) > 0 and field in ret['exprs'][0]:
            return ret['exprs'][0][field]
        else:
            return 0.0

    def _build_fac_exprs(self, expr1: Expression, expr2: Expression) -> str:
        axpr1, axpr2 = Qlib2Wave.waveexpr(expr1), Qlib2Wave.waveexpr(expr2)
        euid = genuid('alpgen.')
        fac1, fac2 = f"{euid}.f1", f"{euid}.f2"
        tfac1 = f"{fac1} <- {axpr1}"
        tfac2 = f"{fac2} <- {axpr2}"
        return [fac1, fac2], [tfac1, tfac2]

    def _build_fac_expr(self, expr1: Expression, index, euid: str) -> str:
        return f"{euid}.f{index} <- {Qlib2Wave.waveexpr(expr1)}"

    def _get_fac_name(self, expr):
        return expr.split(' <- ')[0]

    def _corr(self, x, y):
        valid_index = ~np.isnan(x) & ~np.isnan(y)
        valid_x = x[valid_index]
        valid_y = y[valid_index]
        corr = np.corrcoef(valid_x.reshape((-1)), valid_y.reshape((-1)))[0, 1]
        return np.isnan(corr) and 1.0 or corr

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        facs, exprs = self._build_fac_exprs(expr1, expr2)
        ret = self.wavecalc.rawfactor(exprs)
        corr = self._corr(ret[facs[0]], ret[facs[1]])
        print(f"calc_mutual_IC corr: {corr} {str(expr1)} - {str(expr2)}")
        return corr

    def calc_mutual_ICs(self, expr1: Expression, exprs: List[Expression]) -> List[float]:
        euid = genuid('alpgen.')
        wexprs = [self._build_fac_expr(expr, i+1, euid) for i, expr in enumerate(exprs)]
        wexprs.insert(0, self._build_fac_expr(expr1, 0, euid))
        wfacts = [self._get_fac_name(e) for e in wexprs]
        rets = self.wavecalc.rawfactor(wexprs)
        corrs = [self._corr(rets[wfacts[0]], rets[fac]) for fac in wfacts[1:]]
        return corrs

    def calc_single_IC_ret(self, expr: Expression, custom_vs = None) -> float:
        e = [Qlib2Wave.waveexpr(expr)]
        ret = self.wavecalc.score(e, metrics=['IC'])
        return self._get_result(ret)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self.calc_single_IC_ret(expr)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        ic = self.calc_single_IC_ret(expr)
        return ic, ic

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        # pdb.set_trace()
        euid = genuid('alpgen.')
        wexprs = [f"{euid}.expr.{i} <- {Qlib2Wave.waveexpr(e)}" for i, e in enumerate(exprs)]
        namemap = {n.split(' <- ')[0]: i for i, n in enumerate(wexprs)}

        facs = self.wavecalc.rawfactor(wexprs, burnin = 0) # keep the burns

        samp = facs[list(namemap)[0]]
        wf = KTD(samp.ks, samp.ts, samp.ds, 0.0)
        for n, v in facs.items():
            v[np.isnan(v)] = 0.0
            wf = wf + v * weights[namemap[n]]

        custom_vs = {euid : wf}
        score = self.wavecalc.score([euid], vs=custom_vs, metrics=['IC'])
        return self._get_result(score)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        return self.calc_pool_IC_ret(exprs, weights)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        if len(exprs) <= 0:
            return (0.0, 0.0)
        ic = self.calc_pool_IC_ret(exprs, weights)
        return ic, ic
