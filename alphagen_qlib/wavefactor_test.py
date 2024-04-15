from wavefactor import *

exprs = [
    "Scale(Bound(Neg(TsMean(wdb/ashareyield/PCT_CHANGE_W,50))))",
    "Neg(Div(Broadcast(30.0),Div(Mult(md/std/Vwp,Broadcast(30.0)),md/std/Cls)))"
    ]

drange="20190101-20201220"
wf3 = WaveFactor(date_range=drange, univ="alev", score_expr="ic.ir", threshold=0.1)
af = wf3.factor(exprs)
wf3.save(af, "/home/zyyu/tmp/EOD2/")

score = wf3.score(exprs)
print(score)

metric = wf3.metrics(exprs)
print(metric)
