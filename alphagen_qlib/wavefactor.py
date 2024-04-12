#!/usr/bin/env python3
from collections import defaultdict
import logging
import json
import copy
import numpy as np
import click
import gc
import hashlib
from typing import List, Optional, Tuple
from numpy import ndarray

np.seterr(divide="ignore", invalid="ignore")

from scipy import stats
import matplotlib.pyplot as plt

import os
import sys

libpath = "/home/zyyu/camp/zyyu/repo/gear.v5/ext/whale/blar/env/lib/python3.8/site-packages/"
sys.path.insert(0, os.path.realpath(libpath))
from wavel import *
from legion import *
from tero.cal import *

from whale import Wave, Whale

"""
sample input for scores.py
{
  "legion":"/slice/ljs/cne/EOD",
  "univ": "ZZ800",
  "freq": "EOD",
  "y":"retv225.my",
  "hedge":"hedge/000905.SH",
  "fwd":5,
  "burnin":240,
  "metrics": ["IC","Ret", "Thret", "Tret", "Lhret", "Lret", "Qret"],
  "score":"IC",
  "threshold":0.02,
  "date.range": ["20170401-20191231"],
  "eval.period": ["20170401-1231","20180101-1231","20190101-1231"],
  "exprs": [
    {
      "name": "expr.11",
      "expr": "Scale(Prod(TsEma(Rank(cq/zy1a/ZY1.v84),20),TsStd(cq/zy1a/ZY1.v45,122)))",
    },
      "name": "expr.14",
      "expr": "Scale(TsMedian(Div(mdn/H2PC,mdn/H2V),122))",
    },
    {
      "name": "expr.113",
      "expr": "Scale(DeMean(Log(TsMax(mdn/Vol.ldm5,48))))",
    }
  ]
}
"""
QRET_QUANTILE_NUM = 10
QRET_OUT_ORDER_NUM = [0] * QRET_QUANTILE_NUM
QRET_HIGH_DIFF_SIGN = 0
QRET_LOW_DIFF_SIGN = 0
QRET_TOTAL_EXPR_NUM = 0


def reset_qret_stat():
    global QRET_OUT_ORDER_NUM, QRET_HIGH_DIFF_SIGN, QRET_LOW_DIFF_SIGN, QRET_TOTAL_EXPR_NUM
    QRET_OUT_ORDER_NUM = [0] * QRET_QUANTILE_NUM
    QRET_HIGH_DIFF_SIGN = 0
    QRET_LOW_DIFF_SIGN = 0
    QRET_TOTAL_EXPR_NUM = 0


def build_qret_stat():
    r = {
        "out.of.order": QRET_OUT_ORDER_NUM,
        "high.sign.diff": QRET_HIGH_DIFF_SIGN,
        "low.sign.diff": QRET_LOW_DIFF_SIGN,
        "total.expr.num": QRET_TOTAL_EXPR_NUM,
    }

    return r


def qret_ret_rsquare(ret_ics):
    from scipy import stats

    x = list(range(1, 11))
    y = [r * 10000 for r in ret_ics]
    sl, _, r_value, _, _ = stats.linregress(x, y)
    if not np.isfinite(sl) or not np.isfinite(r_value):
        sl = r_value = 0
    return sl, r_value * r_value


def qret_score_from_ret(ret_irs):
    """
    calculate the quantile returns score
    """
    global QRET_OUT_ORDER_NUM, QRET_HIGH_DIFF_SIGN, QRET_LOW_DIFF_SIGN, QRET_TOTAL_EXPR_NUM
    QRET_TOTAL_EXPR_NUM += 1

    quant_rets = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 0]
    quant_irs = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 1]
    quant_prs = [ret_irs[i] for i in range(0, len(ret_irs)) if i % 3 == 2]
    quant_num = len(quant_rets)

    out_order_num = 0
    for i in range(quant_num - 1):
        if quant_rets[i] > quant_rets[i + 1]:
            out_order_num = out_order_num + 1
            QRET_OUT_ORDER_NUM[i] += 1

    corr1 = np.corrcoef(np.array(quant_rets), np.array(list(range(quant_num))))[0, 1]
    corr2 = np.corrcoef(np.array(quant_rets[1:]), np.array(list(range(quant_num - 1))))[0, 1]
    corr3 = np.corrcoef(np.array(quant_rets[:-1]), np.array(list(range(quant_num - 1))))[0, 1]

    sl, rsquare = qret_ret_rsquare(quant_rets)
    if corr1 * corr2 < 0:
        QRET_LOW_DIFF_SIGN += 1
        return (0, 0, rsquare, sl)
    if corr1 * corr3 < 0:
        QRET_HIGH_DIFF_SIGN += 1
        return (0, 0, rsquare, sl)

    return (quant_rets[-1], quant_irs[-1], rsquare, sl)


def workable_expr(e):
    non_supported_op = [
        "NA",
        "Sigmoid",
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
    scp = sc if sc[-1] == '(' else sc + '('
    pnum = scp.count('(')
    if expr[:len(scp)] == scp:
        return expr
    else:
        return scp + expr + ")"*pnum


def factor_expr(name, expr):
    return f"{name} <- {scale_bound_expr(expr)}"


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
    assert("hedge" in jsn)
    return jsn["hedge"]


def get_fwd_info(jsn):
    assert("fwdexpr" in jsn)
    return jsn["fwdexpr"]


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


@click.command()
@click.argument("eval-file", metavar="<eval.jsn>", type=click.Path(exists=True))
@click.argument("score-file", metavar="<output-score.jsn>", type=click.Path())
def main(eval_file, score_file):
    logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    jsn = read_jsn(eval_file)
    recalc_neg_ic_thresh = "recalc.neg.ic" in jsn and jsn["recalc.neg.ic"] or None
    res = calc_score(jsn, recalc_neg_ic_thresh)
    save_jsn(score_file, res)
    logging.info("Done")


def intersect_train_verf(train_jsn, verf_jsn):
    train_exprs = set([e["name"] for e in train_jsn["exprs"]])
    verf_exprs = set([e["name"] for e in verf_jsn["exprs"]])
    com_exprs = verf_exprs.intersection(train_exprs)
    verf_jsn["exprs"] = [e for e in verf_jsn["exprs"] if e["name"] in com_exprs]
    return verf_jsn


def get_days_count(date_spec):
    if len(date_spec) <= 0:
        return 0

    days = bizdays(date_spec)
    logging.info("drange %s: count %d", date_spec, len(days))
    return len(days)


# verf.drange may be empty string or no key
def get_date_range(jsn):
    if "train.drange" in jsn and "verf.drange" in jsn:
        train_days = bizdays(jsn["train.drange"])
        verf_days = [] if len(jsn["verf.drange"]) <= 0 else bizdays(jsn["verf.drange"])
        last_day = verf_days[-1] if len(verf_days) > 0 else train_days[-1]
        return train_days[0].strftime("%Y%m%d") + "-" + last_day.strftime("%Y%m%d")
    else:
        return jsn["date.range"]


def calc_score(jsn, recalc_neg_ic_thresh):
    """
    "legion":["/cache/ag/legion/cne"],
    "univ": "ZZ800",
    "freq": "DAILY",
    "y":"retv225.my",
    "ypath":"retv225.my/fwd_5/fwd.Retv225.DAILY.5",
    "fwdexpr":"retv225.my/fwd_5/fwd.Retv225.DAILY.5",
    "fwd":5,
    "burnin":240,
    "metrics": ["IC","Aret","Tret"],
    "score":"ic",
    "threshold":0.02,
    "date-range": "20170401-20191231",
    "eval-period": ["20170401-1231","20180101-1231","20190101-1231"],
    """

    # root, date_spec, outfile, univ, freq, fwd, yret, ic, ir, burnin
    # basic things
    root = jsn["legion"]
    univ = jsn["univ"]
    freq = jsn["freq"]
    date_spec = get_date_range(jsn)
    burnin = "burnin" in jsn and jsn["burnin"] or 0
    burnin = burnin > 0 and burnin or (freq == "EOD" and 192 or 21)

    dates = bizdays(date_spec)
    lgn_begin = bizday(dates[0], -burnin).strftime("%Y%m%d")
    lgn_date_spec = lgn_begin + "-" + dates[-1].strftime("%Y%m%d")
    logging.info("Legion root: %s" % (root))
    logging.info("Date spec: %s; burnin: %d; legion: %s" % (date_spec, burnin, lgn_date_spec))
    logging.info(f"Recalc neg ic threshold: {recalc_neg_ic_thresh}")

    idx_path = get_index_path(jsn)
    fwd_expr = get_fwd_info(jsn)
    logging.info("Idx: %s" % (idx_path))
    logging.info("Fwd expr: %s" % (fwd_expr))
    score_expr = "score.expr" in jsn and jsn["score.expr"] or ""
    tse = trans_score_expr(score_expr)
    logging.info(f"Score expr: {score_expr}; translate score expr: {tse}")

    # legion
    logging.info("Open %s as input data directory" % root)
    lgn = Legion(root.split(";"), debug=False)
    logging.debug("Opened")

    loader = lgn.loader(lgn_date_spec, univ=univ, freq=freq)
    na = loader["=na"]

    def load(var):
        if var == "na/na":
            return na
        return loader[var]

    # expr
    rexprs = [
        (i, e["name"], scale_bound_expr(e["expr"])) for i, e in enumerate(jsn["exprs"]) if workable_expr(e["expr"])
    ]  # Workaround: remove exprs with none implemented OP
    logging.info("Read %s exprssions" % len(rexprs))

    # build scoring exprs
    exprs = []
    for m in jsn["metrics"]:
        op = m.replace("+", "")
        # sc.thret/exprname1 <- Thret(Scale(Bound(TsMean(moneyflow/XXXX, 5))), retv225/fwd_5/fwd.Ret.DAILY.5)
        es = [buile_score_expr(op, n, e, jsn["fwd"], fwd_expr, idx_path) for _, n, e in rexprs]
        exprs.extend(es)
    logging.info("IC expressions formatted")

    expr2idx = {}  # expr name to index
    for i, n, _ in rexprs:
        expr2idx[n] = i

    # wave
    graph = Wave()
    shape = tuple(map(len, loader.dims()))
    graph.shape = shape
    graph.build([e for _, e in exprs])
    logging.info("Graph compiled")

    logging.info("Prepare inputs...")
    inputs = {var: load(graph.var_name(var)) for var in graph.vars}
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
    jsn = calc_date_range_score(jsn, tse, expr2idx, shape, length_scores, length_term2inx, burnin)
    logging.info("Metrics computed")

    # Calc Neg(expr) for IC < 0
    if recalc_neg_ic_thresh is not None:
        neg_ic_jsn = copy.deepcopy(jsn)
        neg_ic_jsn["exprs"] = [
            expr for expr in neg_ic_jsn["exprs"] if "ic.mean" in expr and expr["ic.mean"] <= recalc_neg_ic_thresh
        ]
        logging.info(
            "Recalculate the negtive IC exprs number: %d for threshold: %.4f",
            len(neg_ic_jsn["exprs"]),
            recalc_neg_ic_thresh,
        )
        if len(neg_ic_jsn["exprs"]) <= 100:
            logging.info("Too less negtive IC exprs %d, recalculation ignored", len(neg_ic_jsn["exprs"]))
        else:
            for es in neg_ic_jsn["exprs"]:
                es["name"] = f"{es['name']}.neg"
                es["expr"] = f"Neg({es['expr']})"
                if "chromo" in es:
                    es["chromo"] = f"Neg({es['chromo']})"

            reset_qret_stat()
            neg_ic_jsn = calc_score(neg_ic_jsn, None)

            jsn["exprs"].extend(neg_ic_jsn["exprs"])
            jsn["qstat.neg"] = build_qret_stat()
    jsn["exprs"] = sorted(jsn["exprs"], key=lambda x: x["score"], reverse=True)
    logging.info(f"Total {len(jsn['exprs'])} expr returned")

    return jsn


#
def calc_date_range_score(jsn, score_expr, expr2idx, shape, length_scores, length_term2inx, burnin, days=0):
    scs = {}  # the all final result
    # calculate the final scores, and put the scores together
    daybars = get_day_bars(jsn["freq"], jsn["fwd"])
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

        jsn["exprs"][expr_idx][m.lower() + ".mean"] = aic
        jsn["exprs"][expr_idx][m.lower() + ".ir"] = air
        if m == "qret":
            jsn["exprs"][expr_idx][m.lower() + ".slp"] = slp
            jsn["exprs"][expr_idx][m.lower() + ".r2"] = apr

        if expr_idx not in accepted_exprs_idx:
            accepted_exprs_idx.add(expr_idx)
            accepted_exprs.append(jsn["exprs"][expr_idx])

    for expr in accepted_exprs:
        expr["score"] = get_expr_score(expr, score_expr, "ic.ir")

    jsn["exprs"] = [e for e in accepted_exprs if e["score"] >= jsn["threshold"]]
    jsn["qstat"] = build_qret_stat()

    return jsn


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


if __name__ == "__main__":
    main()


def md5str(s, return_bits = 8):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[-return_bits:]


class WaveFactor:
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
        self.calf = Whale(self.jsn['legion'].split(";"), univ=self.jsn['univ'],
            freq=self.jsn['freq'])[self.jsn["daterange"]]
        logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            level=self.jsn['loglevel'])

    def _build_default(self, exprs, **kwargs):
        def kwargs_or(key, value):
            return key in kwargs and kwargs[key] or value

        jsn = {}
        jsn['vid'] = kwargs_or('vid', 10105)
        jsn['univ'] = kwargs_or('univ', 'ZZ800')
        jsn['freq'] = kwargs_or('freq', 'EOD')
        jsn['fwd'] = kwargs_or('fwd', 5)
        jsn['daterange'] = kwargs_or('date_range', '20130101-20201231')
        jsn['threshold'] = kwargs_or('theshhold', 0.01)
        jsn['score.expr'] = kwargs_or('score_expr', 'abs(ic.ir')
        jsn['legion'] = kwargs_or('legion', '/home/zyyu/data/legion/cne/EOD')
        jsn['metrics'] = kwargs_or('metrics', ["IC", "Ret", "Thret", "Qret", "Trv"])
        jsn['recalc.neg.ic'] = kwargs_or('recalc_neg_ic', -0.1)
        jsn['sc'] = kwargs_or('sc', 'Scale(Bound(')
        jsn['fwdexpr'] = kwargs_or('fwdexpr', 'bfwd/Retv225_rt_Retv225/fwd_5')
        jsn['hedge'] = kwargs_or('hedge', 'bfwd/wdIdxEod_md_Ret/fwd_5/000905.SH')
        jsn['exprs'] = [{'name':'expr.'+str(i), 'expr': expr} for i, expr in enumerate(exprs)]
        jsn['loglevel'] = kwargs_or('loglevel', logging.DEBUG)

        return jsn


    def _load_var(self, var):
        if var in self.loaded_vs:
            logging.debug(f"Bypass {var} loading")
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

        idx_path = jsn["hedge"]
        fwd_expr = jsn["fwdexpr"]
        logging.info(f"Index path: {idx_path} Fwd expr: {fwd_expr}")
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


    def factor(self, raw_exprs):
        jsn = self.jsn

        sc = "sc" in jsn and jsn["sc"] or "Scale"
        logging.debug(f"SC method: {sc}")
        jsn['exprs'] = [
                        {'name':'.'.join(['expr', str(i), md5str(expr)]), 'expr': expr}
                        for i, expr in enumerate(raw_exprs)
                       ]
        logging.info(jsn['exprs'])

        rexprs = [
                   (i, e["name"], scale_bound_expr(e["expr"], sc)) 
                   for i, e in enumerate(jsn['exprs']) if workable_expr(e["expr"])
                 ]
        logging.info(f"Read {len(rexprs)} exprssions")

        exprs = [factor_expr(n, e) for _, n, e in rexprs]
        logging.info(f"Factor expressions formatted {len(exprs)}")

        wir = Wave.compile_or_load(exprs)

        return self.calf.eval(wir, burn = self.burnin)


    def save(self, terms, obase):
        dst = Legion(obase, 'w')

        for path, ktd in terms.items():
            dst[path] = ktd


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
        exprs = []
        eval_metrics = metrics if metrics else jsn["metrics"]
        for m in eval_metrics:
            op = m.replace("+", "")
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

        inputs = {var: self._load_var(graph.var_name(var)) for var in graph.vars}
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


    def metrics(self, raw_exprs, metrics = None):
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
        eval_metrics = metrics if metrics else jsn["metrics"]
        for m in eval_metrics:
            op = m.replace("+", "")
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

        inputs = {var: self._load_var(graph.var_name(var)) for var in graph.vars}
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

        for term_name, ktd in outs.items():
            expr_name, metric = expr_metrics_from_name(term_name)
            jsn['exprs'][expr2idx[expr_name]][metric] = ktd[:,:,self.burnin:].flatten(order="F").tolist()

        return jsn

