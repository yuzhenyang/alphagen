from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression

from numpy import ndarray
from typing import List, Optional, Tuple

from legion import *
from tero.cal import *
from wavel import *

import gc
import hashlib
import logging
import numpy as np
import pdb
from collections import defaultdict
import copy
import logging
import json
from scipy import stats
import numbers
from itertools import product


np.seterr(divide="ignore", invalid="ignore")


def qret_ret_rsquare(ret_ics):
    x = list(range(1, 11))
    y = np.array(ret_ics).argsort().argsort() # rank ret_ics
    sl, _, r_value, _, _ = stats.linregress(x, y)
    if not np.isfinite(sl) or not np.isfinite(r_value):
        sl = r_value = 0
    return sl, r_value * r_value


def qret_score_from_ret(ret_irs):
    """
    calculate the quantile returns score
    """

    quant_rets = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 0]
    quant_irs = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 1]
    quant_prs = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 2]
    quant_num = len(quant_rets)

    out_order_num = 0
    for i in range(quant_num - 1):
        if quant_rets[i] > quant_rets[i + 1]:
            out_order_num = out_order_num + 1

    corr1 = np.corrcoef(np.array(quant_rets), np.array(list(range(quant_num))))[0, 1]
    corr2 = np.corrcoef(np.array(quant_rets[1:]), np.array(list(range(quant_num - 1))))[0, 1]
    corr3 = np.corrcoef(np.array(quant_rets[:-1]), np.array(list(range(quant_num - 1))))[0, 1]

    sl, rsquare = qret_ret_rsquare(quant_rets)
    if corr1 * corr2 < 0:
        return (0, 0, rsquare, sl)
    if corr1 * corr3 < 0:
        return (0, 0, rsquare, sl)

    return (quant_rets[-1], quant_irs[-1], rsquare, sl)


def daily_ic(ics):
    if np.count_nonzero(~np.isfinite(ics)) * 5 > ics.size:
        return np.nan
    return float(np.nanmean(ics))


def workable_expr(e):
    non_supported_op = [
        "NA",
        "KRedMean",
        "KRedMed",
        "LmRes(",
        "LmResIntXY",
        "TsAutoCor",
        "TsDeltaCor",
        "TsCoSkew",
        "TsCoVar",
    ]
    for op in non_supported_op:
        if e.find(op) != -1:
            return False
    return True


def scale_bound_expr(expr, sc="Scale"):
    if expr[:6] == (sc + "("):
        return expr
    elif expr[:6] == "Bound(":
        return sc + "(" + expr + ")"
    else:
        return sc + "(Bound(" + expr + "))"


QRET_QUANTILE_NUM = 10
def buile_score_expr(op, n, expr, fwd, fwd_expr, hedge):
    # encoding for result length
    # Qret returns contain QRET_QUANTILE_NUM values
    rlen = op == "Qret" and QRET_QUANTILE_NUM or 1
    vname = "sc.%s.%d./%s" % (op.lower(), rlen, n)

    if op == "Tret":
        r = (vname, "%s <- %s(%s, %s)" % (vname, "Thret", expr, fwd_expr))
    elif op == "Lret":
        r = (vname, "%s <- %s(%s, %s)" % (vname, "Lhret", expr, fwd_expr))
    elif op == "Thret" or op == "Lhret":
        r = (vname, "%s <- %s(%s, %s, %s)" % (vname, op, expr, fwd_expr, hedge))
    elif op == "Trv" or op == "TopTrv":
        r = (vname, "%s <- %s(%s, %d)" % (vname, op, expr, fwd))
    else:
        r = (vname, "%s <- %s(%s, %s)" % (vname, op, expr, fwd_expr))
    return r



def buile_factor_expr(i, expr):
    return r


# ie: sc.qret.10./expr1 ==> 10
#     sc.tret.1./expr1  ==> 1
def result_len_from_name(term_name):
    ps = term_name.split("./")
    if len(ps) < 2:
        return 1
    dot_rindex = ps[0].rfind(".")
    return int(ps[0][dot_rindex + 1 :])


def expr_metrics_from_name(term_name):
    ps = term_name.split("./")
    assert(len(ps) == 2)
    metric = ps[0].split(".")[1]
    expr_name = ps[1]
    return expr_name, metric


def get_index_path(jsn):
    default_index = "bfwd/%s/fwd_%d/000905.SH" % (  # default ZZ500
        "wdIdxEod_md_Ret" if jsn["freq"] == "EOD" else "hqIdx_Ret",
        jsn["fwd"],
    )
    return "hedge" in jsn and jsn["hedge"] or default_index


def get_fwd_info(jsn):
    fwddir = "bfwd/Retv225_rt_Retv225/" if jsn["freq"] == "EOD" else "bfwd/std.Retv2_md_Retv"
    fwd_path = "ypath" in jsn and jsn["ypath"] or "%s/fwd_%d" % (fwddir, jsn["fwd"])
    if "fwdexpr" in jsn:
        fwd_expr = jsn["fwdexpr"]
    else:
        fwd_expr = "Cond(Or(Less(md/std/Hgh,md/lim/up),Greater(md/std/Low,md/lim/dn)),%s,na/na)" % (fwd_path)
    return fwd_expr


def trans_score_expr(score_expr, expr_name="expr"):
    if len(score_expr) <= 0:
        return ""
    comb_fields = ["ic", "thret", "lhret", "ret", "qret", "tret", "lret", "trv", "topic", "toptrv"]
    comb_metrics = ["mean", "ir"]
    singles = ["qret.slp", "qret.r2"]
    all_fm = [f + "." + m for f in comb_fields for m in comb_metrics]
    all_fm.extend(singles)

    def first_whole_matching(s, p, start):
        t = "%" + s + "%"
        idx = t.find(p, start)
        while idx > 0:
            if not t[idx - 1].isalpha() and not t[idx + len(p)].isalpha():
                return True, idx - 1, idx + len(p) - 1
            else:
                idx = t.find(p, idx + len(p))
        else:
            return False, -1, -1

    ret = score_expr.lower()

    for fm in all_fm:
        start = 0
        while True:
            match, l, r = first_whole_matching(ret, fm, start)
            if not match:
                break
            rep = f"{expr_name}['{fm}']"
            ret = ret[:l] + rep + ret[r:]
            start = l + len(rep)

    return ret


def get_expr_score(expr, score_expr, scol):
    try:
        if len(score_expr) > 0:
            return eval(score_expr)
        else:
            return expr[scol]
    except Exception:
        return 0.0


# Get day bar count for final scoring
# M10P6 18 bars, M10P24 23 bars, EOD 1 bar
def get_day_bars(freq, fwd):
    if freq == "M10":
        if fwd == 6:
            return 18
        elif fwd == 24:
            return 23
    elif freq == "EOD" or freq == "DAILY":
        return 1

    logging.error(f"get_day_bars invald freq: {freq}; fwd: {fwd}")
    assert False


def md5str(s, return_bits = 8):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[-return_bits:]


#
def calc_date_range_score(conf, exprs, py_score_expr, expr2idx, shape, length_scores, length_term2inx, burnin, days=0, include_neg_ic = None):
    scs = {}  # the all final result
    # calculate the final scores, and put the scores together
    daybars = get_day_bars(conf["freq"], conf["fwd"])
    logging.info(f"Day bar count: {daybars}")

    for rlen, ess in length_scores.items():
        score_shape = [rlen, shape[1], shape[2]]
        sc = final_scores(ess, score_shape, burnin, daybars, days)
        for term_name, index in length_term2inx[rlen].items():
            scs[term_name] = sc[index]
    logging.info("Final scores computed")

    accepted_exprs = []
    accepted_exprs_idx = set()

    logging.info("Metrics conducting...")
    # check the scoring result
    for vname, r in scs.items():
        m, name = extract_op_vname(vname)
        scores = extract_score(m, r)
        expr_idx = expr2idx[name]
        aic, air, apr, slp = scores
        if np.isnan(aic) or np.isnan(air):
            continue

        exprs[expr_idx][m.lower() + ".mean"] = aic
        exprs[expr_idx][m.lower() + ".ir"] = air
        if m == "qret":
            exprs[expr_idx][m.lower() + ".slp"] = slp
            exprs[expr_idx][m.lower() + ".r2"] = apr

        if expr_idx not in accepted_exprs_idx:
            accepted_exprs_idx.add(expr_idx)
            accepted_exprs.append(exprs[expr_idx])

    for expr in accepted_exprs:
        expr["score"] = get_expr_score(expr, py_score_expr, "ic.ir")
    exprs = [e for e in accepted_exprs if e["score"] >= conf["threshold"]]

    if include_neg_ic:
        neg_ic_exprs = [e for e in accepted_exprs if "ic.mean" in e and e["ic.mean"] <= include_neg_ic]
    else:
        neg_ic_exprs = []

    return exprs, neg_ic_exprs


def extract_op_vname(vname):
    var, name = vname.split("/")
    m = var.split(".")[1]
    return m, name


def extract_score(m, r):
    oon = 0
    if m == "qret":
        aic, air, pr, oon = qret_score_from_ret(r)
    else:
        aic, air, pr = r
    return aic, air, pr, oon


def is_number(x):
    return isinstance(x, numbers.Number)



class WaveCalculator(AlphaCalculator):
    lgn = None
    loader = None
    graph = None
    jsn = None
    burnin = 192
    loglvl = logging.ERROR
    loaded_vs = {}

    ready = False

    def __init__(self, **kwargs):
        self.jsn = self._build_default([], **kwargs)
        logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=self.jsn['loglvl'])

    def _build_default(self, exprs, **kwargs):
        def kwargs_or(key, value):
            return key in kwargs and kwargs[key] or value

        jsn = {}
        jsn['vid'] = kwargs_or('vid', 10105)
        jsn['univ'] = kwargs_or('univ', 'ZZ800')
        jsn['freq'] = kwargs_or('freq', 'EOD')
        jsn['fwd'] = kwargs_or('fwd', 5)
        jsn['train.drange'] = kwargs_or('date_range', '20130101-20201231')
        jsn['verf.drange'] = kwargs_or('verf_date', '')
        jsn['threshold'] = kwargs_or('theshhold', 0.01)
        jsn['score.expr'] = kwargs_or('score_expr', 'abs(ic.ir')
        jsn['legion'] = kwargs_or('legion', '/home/zyyu/data/legion/cne/EOD')
        jsn['metrics'] = kwargs_or('metrics', ["IC", "Ret+", "Thret+", "Qret+", "Trv+"])
        jsn['recalc.neg.ic'] = kwargs_or('recalc_neg_ic', -0.1)
        jsn['sc'] = kwargs_or('sc', 'Scale')
        jsn['fwdexpr'] = kwargs_or('fwdexpr', 'bfwd/Retv225_rt_Retv225/fwd_5')
        jsn['hedge'] = kwargs_or('hedge', 'bfwd/wdIdxEod_md_Ret/fwd_5/000905.SH')
        jsn['exprs'] = [{'name':'expr.'+str(i), 'expr': expr} for i, expr in enumerate(exprs)]
        jsn['log_level'] = kwargs_or('loglvl', logging.DEBUG)


    def _load_var(self, var):
        if var in self.loaded_vs:
            logging.trace(f"Bypass {var} loading")
            return self.loaded_vs[var]

        self.loaded_vs[var] = self.loader[var]
        return self.loaded_vs[var]


    def _unload_var(self, var):
        if var in self.loaded_vs:
            del self.loaded_vs[var]


    def _prepare(self):
        if self.ready:
            return self.jsn

        jsn = self.jsn
        root = jsn["legion"]
        univ = jsn["univ"]
        freq = jsn["freq"]
        date_spec = get_date_range(jsn)
        burnin = "burnin" in jsn and jsn["burnin"] or 0
        burnin = burnin > 0 and burnin or (freq == "EOD" and 192 or 21)

        self.burnin = burnin

        dates = bizdays(date_spec)
        lgn_begin = bizday(dates[0], -burnin).strftime("%Y%m%d")
        lgn_date_spec = lgn_begin + "-" + dates[-1].strftime("%Y%m%d")
        logging.info(f"Legion root: {root}; univ: {univ}; freq: {freq}")
        logging.info(f"Date spec: {date_spec}; burnin: {burnin}; legion: {lgn_date_spec}")

        idx_path = get_index_path(jsn)
        fwd_expr = get_fwd_info(jsn)
        logging.info(f"Index path: {idx_path}")
        logging.info(f"Fwd expr: {fwd_expr}")
        score_expr = "score.expr" in jsn and jsn["score.expr"] or ""
        tse = trans_score_expr(score_expr)
        logging.info(f"Score expr: {score_expr}; translate score expr: {tse}")


        self.lgn = Legion(root.split(";"), debug=False)
        logging.debug(f"Legion root {root} opened")

        self.loader = self.lgn.loader(lgn_date_spec, univ=univ, freq=freq)

        self.loaded_vs["na/na"] = self.loader["=na"]

        self.graph = Wave()
        shape = tuple(map(len, self.loader.dims()))
        logging.info(f"Data dimension: {str(shape)}")
        self.graph.shape = shape

        self.ready = True
        return self.jsn


    def score(self, raw_exprs, metrics = None):
        jsn = self._prepare()

        sc = "sc" in jsn and jsn["sc"] or "Scale"
        logging.debug(f"SC method: {sc}")
        jsn['exprs'] = [{'name':'.'.join(['expr', str(i), md5str(expr)]), 'expr': expr} for i, expr in enumerate(raw_exprs)]
        logging.info(jsn['exprs'])

        rexprs = [
            (i, e["name"], scale_bound_expr(e["expr"], sc)) for i, e in enumerate(jsn['exprs']) if workable_expr(e["expr"])
        ]
        logging.info(f"Read {len(rexprs)} exprssions")

        # build scoring exprs
        exprs = [buile_factor_expr(n, e) for _, n, e in rexprs]
        logging.info(f"Factor expressions formatted {len(exprs)}")

        expr2idx = {}  # expr name to index
        for i, n, _ in rexprs:
            expr2idx[n] = i

        graph = self.graph

        graph.build([e for _, e in exprs])
        logging.info("Graph compiled")

        logging.info("Prepare inputs...")
        shape = self.graph.shape

        inputs = {var: self.load(graph.var_name(var)) for var in graph.vars}
        term_names = [e.split("<-")[0].strip() for _, e in exprs]
        outs = {term_name: np.ndarray(shape, order="F") for term_name in term_names}

        for var, buf in inputs.items():
            graph.set_input(var, buf)

        for term_name, buf in outs.items():
            graph.set_output(graph.fac_node(term_name), buf, buf.shape[0])
        logging.info("Inputs done")

        logging.info("Computing factors...")
        graph.run()
        logging.info("Factors computed")

        return outs


    def factor(self, raw_exprs) -> List(ndarray):
        jsn = self._prepare()

        sc = "sc" in jsn and jsn["sc"] or "Scale"
        logging.debug(f"SC method: {sc}")
        jsn['exprs'] = [{'name':'.'.join(['expr', str(i), md5str(expr)]), 'expr': expr} for i, expr in enumerate(raw_exprs)]
        logging.info(jsn['exprs'])

        rexprs = [
            (i, e["name"], scale_bound_expr(e["expr"], sc)) for i, e in enumerate(jsn['exprs']) if workable_expr(e["expr"])
        ]
        logging.info(f"Read {len(rexprs)} exprssions")

        # build scoring exprs
        exprs = []
        only_positive = {}  # only positive value for result
        eval_metrics = metrics if metrics else jsn["metrics"]
        for m in eval_metrics:
            op = m.replace("+", "")
            only_positive[op.lower()] = m.find("+") > 1  # only positive value is fine
            # sc.thret/exprname1 <- Thret(Scale(Bound(TsMean(moneyflow/XXXX, 5))), retv225/fwd_5/fwd.Ret.DAILY.5)
            es = [buile_score_expr(op, n, e, jsn["fwd"], fwd_expr, idx_path) for _, n, e in rexprs]
            exprs.extend(es)
        logging.info("IC expressions formatted")

        expr2idx = {}  # expr name to index
        for i, n, _ in rexprs:
            expr2idx[n] = i

        logging.info(f"expr2idx:{expr2idx}")

        # wave
        graph = self.graph

        graph.build([e for _, e in exprs])
        logging.info("Graph compiled")

        logging.info("Prepare inputs...")
        shape = self.graph.shape

        inputs = {var: self.load(graph.var_name(var)) for var in graph.vars}
        term_names = [e.split("<-")[0].strip() for _, e in exprs]
        outs = {
            term_name: np.ndarray((result_len_from_name(term_name), shape[1], shape[2]), order="F")
            for term_name in term_names
        }

        for var, buf in inputs.items():
            graph.set_input(var, buf)

        for term_name, buf in outs.items():
            graph.set_output(graph.fac_node(term_name), buf, buf.shape[0])
        logging.info("Inputs done")

        logging.info("Computing daily scores...")
        graph.run()
        logging.info("Score computed")

        # put same result length expr together and calculate the final_score() group by result length
        length_scores = defaultdict(list)  # length : score_ktd
        length_idx = defaultdict(int)  # length : current_index
        length_term2inx = defaultdict(lambda: defaultdict(int))  # length : {term_name:index}

        for term_name, ktd in outs.items():
            rlen = result_len_from_name(term_name)
            length_scores[rlen].append(ktd[:rlen, :, :].flatten(order="F").tolist())
            length_term2inx[rlen][term_name] = length_idx[rlen]
            length_idx[rlen] += 1

        del outs
        gc.collect()

        logging.info("Computing final scores...")
        eval_exprs, neg_ic_exprs = calc_date_range_score(jsn, exprs, tse, expr2idx,
                shape, length_scores, length_term2inx, self.burnin, 0)

        logging.info("Metrics computed")
        return eval_exprs


    def _calc_alpha(self, expr: Expression) -> ndarray:
        # return normalize_by_day(expr.evaluate(self.data))
        return 0.0

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

    # def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
    #     value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
    #     return self._calc_IC(value1, value2)

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