##########################  -*- coding: us-ascii -*- ###########################
# whale --- Wave on Legion
################################################################################
if __file__[-23:] == 'whale/whale/__init__.py':
    import sys
    sys.path = [__file__[:-23] + x for x in ['pylegion', 'pywave']] + sys.path

from legion import *
import wavel, os, math

__all__ = ['Wave', 'Whale', 'Legion', 'KTD', 'NA', 'isNA']

class Wave(wavel.Wave):
    def __init__(self, loader, **kw):
        wavel.Wave.__init__(self)
        self.score = kw.get('score', 0)
        self.max_parallelism = kw.get('max_parallelism', 0)
        self.loader = loader
        dims = loader.dims()
        self.shape = tuple(map(len, dims))
        if len(self.shape) != 3:
            raise ValueError('invalid shape: len(self.shape) != 3')
        self.dims = (['score'], dims[1], dims[2]) if self.score else dims

    def __call__(self, exprs):
        if self.done:
            raise RuntimeError('NO re-run, please define a new Wave')

        wir = self.compile_or_load(exprs)

        # build nodes (side effect: self.vars and self.facs are also built)
        self.build(wir)

        # setup input buffers according to self.vars
        inputs = {}
        for var in self.vars:
            # load KTD from legion
            # inputs[var] = self.loader[self.var_name(var)]
            inputs[var] = self._load_var(self.var_name(var))
            self.set_input(var, inputs[var])

        # setup output buffers according to self.facs
        ans = {}
        for fac in self.facs:
            name = self.fac_name(fac)
            if self.score > 0:
                dim0 = [f"{name}.{i+1}" for i in range(self.score)]
            else:
                dim0 = self.dims[0]
            ans[name] = KTD( dim0, self.dims[1], self.dims[2])
            self.set_output(fac, ans[name])

        # run wave and return results
        self.run(max_parallelism=self.max_parallelism)

        return ans

    def _load_var(self, var):
        if var not in self.loaded_vars:
            self.loaded_vars[var] = self.loader[var]
        return self.loaded_vars[var]

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

class Whale():
    def __init__(self, src, dst = None, **kw):
        # input (src) and output (dst) legion base(s)
        if isinstance(src, Legion):
            self.src = src
        else:
            mode = 'w' if dst is None and self._writable(src) else 'r'
            self.src = Legion(src, mode, **kw)
        if dst is None:
            self.dst = self.src
        elif isinstance(dst, Legion):
            self.dst = dst
        else:
            self.dst = Legion(dst, 'w',
                              univ = self.src.univ,
                              freq = self.src.freq)

    @staticmethod
    def _writable(p):
        return os.access(p if isinstance(p, str) else p[0], os.W_OK)

    # calf = whale['yyyymmdd-mmdd']
    def __getitem__(self, span):
        return Calf(self, span)

    # whale['/path/to/data'] = ktd
    def __setitem__(self, path, ktd):
        self.dst.save(ktd, path)

    def __contains__(self, path):
        return path in self.src

    def bizday(self, dnum, shift = 0):
        return self.src.bizday(dnum, shift)

    def bizdays(self, *args):
        return self.src.bizdays(*args)

    def __str__(self):
        spec = f"'{self.src.root}'"
        if self.src.layers:
            spec = f'[{spec}, ...]'
        return ("Whale(%s, univ='%s', freq='%s')"
                % (spec, self.src.univ, self.src.freq))

    def __repr__(self):
        return str(self)

class Calf():
    def __init__(self, whale, span):
        self.whale  = whale
        self.span   = span
        self.loader = whale.src[span]

    def __contains__(self, path):
        return path in self.loader

    # calf(exprs, burn = ...)
    def __call__(self, exprs, *, burn = -1):
        "Syntax sugar for data loading"
        if isinstance(exprs, dict):
            return self.load(exprs, burn = burn)
        if isinstance(exprs, list):
            # NOTE: requires ordered dict, ONLY works with Python >= 3.7
            edict = {f'id{n}': x for n, x in enumerate(exprs)}
            return list(self.load(edict, burn = burn).values())
        if isinstance(exprs, str):
            return next(iter(self.load({'x': exprs}, burn = burn).values()))
        raise TypeError('invalid exprs type')

    # calf[exprs]
    def __getitem__(self, exprs):
        return self(exprs)

    # calf << exprs
    def __lshift__(self, exprs):
        self.save(self.eval(exprs))

    # exprs >> calf
    def __rrshift__(self, exprs):
        self.results = self.eval(exprs)
        return self

    # calf >> legion
    def __rshift__(self, dst):
        self.save(self.results, dst)
        del self.results

    # try to build the graph
    def validate(self, exprs):
        wir = Wave.compile_or_load(exprs)
        graph = Wave(self.loader)
        graph.build(wir)
        if len(graph.vars) != len(wir.operands):
            raise RuntimeError('len(graph.vars) != len(wir.operands)')
        if len(graph.facs) != len(wir.factors):
            raise RuntimeError('len(graph.facs) != len(wir.factors)')
        return True

    # compute full expressions (with assignment), return result(s)
    def eval(self, exprs, *, burn = -1, max_parallelism = 0):
        # load WIR
        wir = Wave.compile_or_load(exprs)

        if burn < 0:
            # get burn off days from WIR, could still be zero
            burn_days = math.ceil(wir.burn_periods / self.loader.shape[1])
        else:
            burn_days = burn

        if burn_days == 0:
            return Wave(self.loader, max_parallelism=max_parallelism)(wir)

        # first extend, then burn off
        beg = self.bizday(self.loader.beg, -burn_days)
        return {name: v(ds = [burn_days, None]) for name, v in
                Wave(self.loader.span(beg, self.loader.end), max_parallelism=max_parallelism)(wir).items()}

    #  compute semi-expressions (without assignment), return result(s)
    def load(self, exprs_dict, *, burn = -1):
        # transform exprs without names
        exprs = ['%s<-%s' % (name, '(' in expr and expr or f'Self({expr})')
                 for name, expr in exprs_dict.items()]
        return self.eval(exprs, burn = burn)

    def save(self, ktd_dict, dst = None):
        if dst is None:
            dst = self.whale.dst
        elif isinstance(dst, str):
            dst = Legion(dst, 'w')
        elif not isinstance(dst, Legion):
            raise TypeError('invalid dst type')
        for path, ktd in ktd_dict.items():
            dst[path] = ktd

    def bizday(self, dnum, shift = 0):
        return self.whale.bizday(dnum, shift)

    def bizdays(self, *args):
        return self.whale.bizdays(*args)

    def __str__(self):
        return f"{self.whale}['{self.span}']"

    def __repr__(self):
        return str(self)

### whale/__init__.py ends here
