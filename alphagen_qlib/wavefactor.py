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
import pdb

libpath = "/home/zyyu/camp/zyyu/repo/gear.v5/ext/whale/blar/env/lib/python3.8/site-packages/"
sys.path.insert(0, os.path.realpath(libpath))
libpath = "/usr/local/lib/python3.8/dist-packages/"
sys.path.insert(0, os.path.realpath(libpath))

print(sys.path)

from wavel import *
from legion import *
from tero.cal import *
import wavel

QRET_QUANTILE_NUM = 10


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

    # jsn["exprs"] = [e for e in accepted_exprs if e["score"] >= jsn["threshold"]]

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


def datespan(daterange, burn):
    dates = bizdays(daterange)
    begin = bizday(dates[0], -burn).strftime("%Y%m%d")
    return begin + "-" + dates[-1].strftime("%Y%m%d")


class LegionVarLoader:
    def __init__(self, root, univ, freq, daterange, burnin):
        self.lgn = Legion(root.split(";"), univ = univ, freq = freq)
        span = datespan(daterange, burnin)
        self.loader = self.lgn[span]
        self.loaded_vars = {}
        self.custom_vars = {}

    def add_var(self, var, v):
        if var not in self.custom_vars:
            self.custom_vars[var] = v

    def del_var(self, var):
        if var in self.custom_vars:
            del self.custom_vars[var]

    def add_vars(self, vars):
        if not vars:
            return
        for n, v in vars.items():
            logging.debug(f"Add custom var {n}")
            self.custom_vars[n] = v

    def del_vars(self, vars):
        if not vars:
            return
        for n in vars.keys():
            logging.debug(f"Del custom var {n}")
            del self.custom_vars[n]

    def get_var(self, var):
        if var in self.custom_vars:
            logging.debug(f"Loading custom var {var}")
            return self.custom_vars[var]

        if var not in self.loaded_vars:
            logging.debug(f"Loading var: {var}")
            self.loaded_vars[var] = self.loader[var]
        else:
            logging.debug(f"Bypass var loading : {var}")
        return self.loaded_vars[var]

    def dims(self):
        return self.loader.dims()


class Wave(wavel.Wave):
    def __init__(self, loader, **kw):
        wavel.Wave.__init__(self)
        # self.score = kw.get('score', False)
        self.max_parallelism = kw.get('max_parallelism', 0)
        self.loader = loader
        self.dims = loader.dims()
        self.shape = tuple(map(len, self.dims))
        if len(self.shape) != 3:
            raise ValueError('invalid shape: len(self.shape) != 3')

    def eval(self, exprs, burn = 0, score = False):
        wir = self.compile_or_load(exprs)

        # build nodes (side effect: self.vars and self.facs are also built)
        self.build(wir)

        # setup input buffers according to self.vars
        inputs = {}
        for var in self.vars:
            # load KTD from legion
            inputs[var] = self.loader.get_var(self.var_name(var))
            self.set_input(var, inputs[var])

        # setup output buffers according to self.facs
        ans = {}
        for fac in self.facs:
            name = self.fac_name(fac)
            dim0 = score and [f"{name}.{i+1}" for i in range(Wave.rlen_by_name(name))] or self.dims[0]
            ans[name] = KTD(dim0, self.dims[1], self.dims[2])
            self.set_output(fac, ans[name], len(dim0))

        # run wave and return results
        self.run(max_parallelism=self.max_parallelism)

        return {name: v(ds = [burn, None]) for name, v in ans.items()}

    @staticmethod
    def rlen_by_name(term_name):
        return result_len_from_name(term_name)

    @staticmethod
    def compile(exprs):
        "Compile expressions, return WIR"
        if not exprs:
            raise ValueError("compiling empty expression(s)")
        ctx = wavel.Context()
        if isinstance(exprs, str):
            wavel.execute(exprs, ctx)
        else:
            for expr in exprs:
                wavel.execute(expr, ctx)
        return wavel.WIR(ctx)

    @staticmethod
    def load(fname):
        "Load wave file (compiled or not), return WIR"
        if not os.path.exists(fname):
            raise RuntimeError(fname + ' not found!')
        if not os.access(fname, os.W_OK):
            raise RuntimeError(fname + ' not readable!')
        return wavel.WIR(fname)

    @staticmethod
    def compile_or_load(exprs):
        if isinstance(exprs, wavel.WIR):
            # loaded bytecode
            return exprs
        elif isinstance(exprs, str):
            if '<-' in exprs:
                # single expr
                return Wave.compile(exprs)
            else:
                # bytecode file
                return Wave.load(exprs)
        elif isinstance(exprs, list):
            # list[expr]
            return Wave.compile(exprs)
        else:
            raise TypeError("invalid type")


class WaveFactor:
    def __init__(self, **kwargs):
        jsn = self._build_default([], **kwargs)
        self.jsn = jsn
        self.loader = LegionVarLoader(jsn['legion'], jsn['univ'],
                        jsn['freq'], jsn['daterange'], jsn['burn'])
        self.wave = Wave(self.loader)

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
        jsn['legion'] = kwargs_or('legion', '/slice/ljs/cne/EOD')
        jsn['metrics'] = kwargs_or('metrics', ["IC", "Ret", "Thret", "Qret", "Trv"])
        jsn['recalc.neg.ic'] = kwargs_or('recalc_neg_ic', -0.1)
        jsn['sc'] = kwargs_or('sc', 'Scale(Bound(')
        jsn['fwdexpr'] = kwargs_or('fwdexpr', 'bfwd/Retv225_rt_Retv225/fwd_5')
        jsn['hedge'] = kwargs_or('hedge', 'bfwd/wdIdxEod_md_Ret/fwd_5/000905.SH')
        jsn['exprs'] = [{'name':'expr.'+str(i), 'expr': expr} for i, expr in enumerate(exprs)]
        jsn['loglevel'] = kwargs_or('loglevel', logging.DEBUG)

        burnin = "burnin" in jsn and jsn["burnin"] or 0
        burnin = burnin > 0 and burnin or (jsn['freq'] == "EOD" and 192 or 21)
        jsn['burn'] = burnin

        return jsn


    def factor(self, raw_exprs, burnin = -1, vs = None):
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
        burn = burnin < 0 and self.jsn['burn'] or burnin

        self.loader.add_vars(vs)
        res = self.wave.eval(exprs, burn = burn)
        self.loader.del_vars(vs)

        return res

    def rawfactor(self, exprs, burnin = -1, vs = None):
        logging.debug(f"Raw factor {len(exprs)}")
        logging.debug(f"Expr {exprs[0]}")
        assert('<-' in exprs[0])

        burn = burnin < 0 and self.jsn['burn'] or burnin

        self.loader.add_vars(vs)
        res = self.wave.eval(exprs, burn = burn)
        self.loader.del_vars(vs)

        return res

    def save(self, terms, obase):
        dst = Legion(obase, 'w')

        for path, ktd in terms.items():
            dst[path] = ktd

    def score(self, raw_exprs, evaly = True, metrics = None, vs = None):
        jsn = self.jsn

        expr2idx = {}  # expr name to index
        if evaly:
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
                es = [buile_score_expr(op, n, e, jsn["fwd"], self.jsn['fwdexpr'], self.jsn['hedge']) for _, n, e in rexprs]
                exprs.extend(es)
            for i, n, _ in rexprs:
                expr2idx[n] = i
        else:
            exprs = raw_exprs
            for i, e in enumerate(exprs):
                sps = e.split(' <- ')
                if len(sps) > 0:
                    expr2idx[sps[0]] = i

        logging.info(f"Expr2idx:{expr2idx}")
        logging.info(f"Building expr2idx:{len(expr2idx)}")
        logging.info("IC expressions formatted")
        # pdb.set_trace()
        logging.info(exprs)


        self.loader.add_vars(vs)
        wir = Wave.compile_or_load([e[1] for e in exprs])
        outs = self.wave.eval(wir, score = True, burn = self.jsn['burn'])
        self.loader.del_vars(vs)
        logging.info("Score computed")

        # put same result length expr together and calculate the final_score() group by result length
        length_scores = defaultdict(list)  # length : score_ktd
        length_idx = defaultdict(int)  # length : current_index
        length_term2inx = defaultdict(lambda: defaultdict(int))  # length : {term_name:index}

        shape = None
        for term_name, ktd in outs.items():
            rlen = result_len_from_name(term_name)
            length_scores[rlen].append(ktd[:rlen, :, :].flatten(order="F").tolist())
            length_term2inx[rlen][term_name] = length_idx[rlen]
            length_idx[rlen] += 1
            if not shape:
                shape = ktd.shape

        del outs
        gc.collect()

        logging.info("Computing final scores...")
        score_expr = "score.expr" in jsn and jsn["score.expr"] or ""
        tse = trans_score_expr(score_expr)
        logging.info(f"Score expr: {score_expr}; translate score expr: {tse}")

        jsn = calc_date_range_score(jsn, tse, expr2idx, shape, length_scores, length_term2inx, self.jsn['burn'])

        logging.info("Metrics computed")
        logging.info(jsn)
        return jsn

    def metrics(self, raw_exprs, metrics = None, vs = None):
        jsn = self.jsn

        sc = "sc" in jsn and jsn["sc"] or "Scale"
        logging.debug(f"SC method: {sc}")
        jsn['exprs'] = [{'name':'.'.join(['expr', str(i), md5str(expr)]), 'expr': expr} for i, expr in enumerate(raw_exprs)]
        # logging.info(jsn['exprs'])

        rexprs = [
            (i, e["name"], scale_bound_expr(e["expr"], sc)) for i, e in enumerate(jsn['exprs']) if workable_expr(e["expr"])
        ]
        logging.info(f"Read {len(rexprs)} exprssions")

        # build scoring exprs
        exprs = []
        eval_metrics = metrics if metrics else jsn["metrics"]
        for m in eval_metrics:
            op = m.replace("+", "")
            es = [buile_score_expr(op, n, e, jsn["fwd"], jsn['fwdexpr'], jsn['hedge']) for _, n, e in rexprs]
            exprs.extend(es)
        logging.info("IC expressions formatted")

        expr2idx = {}  # expr name to index
        for i, n, _ in rexprs:
            expr2idx[n] = i

        logging.info(f"expr2idx:{expr2idx}")
        self.loader.add_vars(vs)
        wir = Wave.compile_or_load([e[1] for e in exprs])
        outs = self.wave.eval(wir, score = True, burn = self.jsn['burn'])
        self.loader.del_vars(vs)
        logging.info("Score computed")

        for term_name, ktd in outs.items():
            expr_name, metric = expr_metrics_from_name(term_name)
            jsn['exprs'][expr2idx[expr_name]][metric] = ktd[:,:,self.jsn['burn']:].flatten(order="F").tolist()

        return jsn