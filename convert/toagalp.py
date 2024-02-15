#!/usr/bin/python3
# -*- coding: us-ascii -*-

import json
import logging
import re
import os
import sys
import click
import pdb
import csv
import re


ops = "$low $close $volume $high $open $vwap".split(' ')
lops = "md/std/Low md/std/Cls md/std/Vol md/std/Hgh md/std/Opn md/std/Vwp".split(' ')

origs = "Mul Sub Ref Mean Med Sum Std Var Max Min Mad Delta WMA EMA Cov Corr Constant".split(' ')
repls = "Mult Minus TsDelay TsMean TsMedian TsSum TsStd TsVar TsMax TsMin TsMad TsDelta TsWma TsEma TsCov TsCorr Broadcast".split(' ')

assert(len(origs) == len(repls))
assert(len(ops) == len(lops))

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)


def loginfo(str):
    logging.info(str)


wdbtables = "AIndexValuation ASHAREBALANCESHEET ASHARECASHFLOW ASHARECONSENSUSROLLINGDATA_CAGR ASHARECONSENSUSROLLINGDATA_FTTM ASHARECONSENSUSROLLINGDATA_FY0 ASHARECONSENSUSROLLINGDATA_FY1 ASHARECONSENSUSROLLINGDATA_FY2 ASHARECONSENSUSROLLINGDATA_FY3 ASHARECONSENSUSROLLINGDATA_YOY ASHARECONSENSUSROLLINGDATA_YOY2 ASHAREEODDERIVATIVEINDICATOR ASHAREFINANCIALINDICATOR ASHAREINCOME ASHAREMARGINTRADE ASHAREMONEYFLOW AShareEnergyindex AShareEnergyindexADJ AShareHolderNumber AShareL2Indicators AShareTechIndicators AShareYield AShareswingReversetrend AShareswingReversetrendADJ Ashareintensitytrend AshareintensitytrendADJ"
wdbmaps = {w.lower(): w for w in wdbtables.split(' ')}

def handle_wdb(e):
    e = e.group(0)
    if not e.startswith('$wdb'):
        return e
    sep = e.split('_')
    # pdb.set_trace()
    return '/'.join(['wdb', wdbmaps[sep[1]], '_'.join(sep[2:]).upper()])


def myexpr(e):
    for o, r in zip(origs, repls):
        e = e.replace(o+"(", r+"(")

    for o, l in zip(ops, lops):
        e = e.replace(o, l)

    pattern = r"\$(.*?),"
    pattern = r"\$(.*?)(?=\W|$)"
    e = re.sub(pattern, handle_wdb, e)

    e = f"Scale(Bound({e}))"

    return e


def tojsn(alpha):
    with open(alpha) as f:
        jsn = json.load(f)

    exprs = jsn["exprs"]
    weight = jsn['weights']

    newexprs = []
    exprspool = set()
    for i in range(len(exprs)):
        e = myexpr(exprs[i])
        if e in exprspool:
            continue

        exprspool.add(e)
        expr = {"id": i+1, "name":f"alpha500.expr.{i+1}", "expr":e, "coef":weight[i]}
        newexprs.append(expr)

    newexprs = sorted(newexprs, key=lambda x: abs(x['coef']), reverse=True)
    doc = {"exprs":newexprs, "vid":10101}
    fn = alpha.replace('.jsn', '.ag.jsn')
    fn = alpha.replace('.json', '.ag.jsn')
    loginfo(fn)
    loginfo(f"{len(newexprs)}")
    with open(fn, "w") as f:
        json.dump(doc, f, indent=4)


@click.command()
@click.argument(
    "input-alphas",
    metavar="<input-alpha.jsn> ...",
    type=click.Path(exists=True, file_okay=True),
    nargs=-1,
)
def main(input_alphas):
    for i in range(0, len(input_alphas)):
        loginfo(input_alphas[i])
        tojsn(input_alphas[i])


if __name__ == "__main__":
    main()

# tojsn('alpha500.jsn')

