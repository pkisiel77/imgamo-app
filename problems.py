import numpy as np
from numba import cfunc, types, carray
import ctypes
import functools


# base class for problem
class Problem(object):
    def __init__(self, nvars, nobjs, bounds, need_repair=True):
        super(Problem, self).__init__()
        self.nvars = nvars
        self.nobjs = nobjs
        self.bounds = bounds
        self.need_repair = need_repair
        self.evaluation_count = np.zeros(nobjs)

    # evaluate all objective functions
    def evaluate_all(self, solutions):
        #self.evaluation_count += solutions.shape[0]
        pass

    # evaluate selected objective function
    def evaluate_one(self, solutions, i):
        #self.evaluation_count[i] += solutions.shape[0]
        pass

    def repair(self, solutions):
        return solutions


doublep = ctypes.POINTER(ctypes.c_double)
c_sig = types.void(types.CPointer(types.double), types.CPointer(types.double), types.intc, types.intc, types.intc)


@cfunc(c_sig)
def _evaluate_one(in_, out, n, m, i):
    solutions = carray(in_, (n, m))
    out_array = carray(out, (n,))
    if i == 0:
        out_array[:] = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(solutions[:, :-1] ** 2 + solutions[:, 1:] ** 2)), axis=1)[:]
    if i == 1:
        out_array[:] = np.sum(np.abs(solutions) ** 0.8 + 5.0 * np.sin(solutions ** 3), axis=1)[:]


class Kursawe(Problem):
    def __init__(self, nvars=3, nobjs=2, bounds=((-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)), need_repair=False):
        super(Kursawe, self).__init__(nvars, nobjs, bounds, need_repair)

    def evaluate_all(self, solutions):
        f = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(solutions[:, :-1]**2 + solutions[:, 1:]**2)), axis=1)
        g = np.sum(np.abs(solutions)**0.8 + 5.0 * np.sin(solutions**3), axis=1)
        return np.stack([f, g], axis=1)

    def evaluate_one(self, solutions, i):
        if i == 0:
            return np.sum(-10.0 * np.exp(-0.2 * np.sqrt(solutions[:, :-1]**2 + solutions[:, 1:]**2)), axis=1)
        if i == 1:
            return np.sum(np.abs(solutions)**0.8 + 5.0 * np.sin(solutions**3), axis=1)

    def evaluate_one_c(self, solutions, i):
        n, m = solutions.shape
        evaluated = np.zeros(n)
        addr_in = solutions.ctypes.data
        addr_out = evaluated.ctypes.data
        ptr_in = ctypes.cast(addr_in, doublep)
        ptr_out = ctypes.cast(addr_out, doublep)
        _evaluate_one.ctypes(ptr_in, ptr_out, n, m, i)
        return evaluated


class DTLZ2(Problem):
    def __init__(self, nobjs=3, nvars=9, need_repair=False):
        nvars = nobjs + nvars
        bounds = ((0, 1),) * nvars
        super(DTLZ2, self).__init__(nvars, nobjs, bounds, need_repair)

    def evaluate_all(self, solutions):
        k = self.nvars - self.nobjs + 1
        g = np.sum(np.power(solutions[:, -k:] - 0.5, 2.0), axis=1)
        f = [1.0 + g.copy() for _ in range(self.nobjs)]
        for i in range(self.nobjs):
            for j in range(self.nobjs - 1 - i):
                f[i] *= np.cos(0.5 * np.pi * solutions[:, j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * solutions[:, self.nobjs - i - 1])
        return np.stack(f, axis=1)

    def evaluate_one(self, solutions, i):
        k = self.nvars - self.nobjs + 1
        g = np.sum(np.power(solutions[:, -k:] - 0.5, 2.0), axis=1)
        f = [1.0 + g.copy() for _ in range(self.nobjs)]

        for i in range(self.nobjs):
            for j in range(self.nobjs - 1 - i):
                f[i] *= np.cos(0.5 * np.pi * solutions[:, j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * solutions[:, self.nobjs - i - 1])
        return f[i]


class DTLZ3(Problem):
    def __init__(self, nobjs=3, nvars=9, need_repair=False):
        nvars = nobjs + nvars
        bounds = ((0.001, 0.999),) * nvars
        super(DTLZ3, self).__init__(nvars, nobjs, bounds, need_repair)

    def evaluate_all(self, solutions):
        k = self.nvars - self.nobjs + 1
        g = 100 * (k + np.sum(np.power(solutions[:, -k:] - 0.5, 2.0)-np.cos(20*np.pi*(solutions[:, -k:] - 0.5)), axis=1))
        f = [1.0 + g.copy() for _ in range(self.nobjs)]
        for i in range(self.nobjs):
            for j in range(self.nobjs - 1 - i):
                f[i] *= np.cos(0.5 * np.pi * solutions[:, j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * solutions[:, self.nobjs - i - 1])
        return np.stack(f, axis=1)

    def evaluate_one(self, solutions, i):
        k = self.nvars - self.nobjs + 1
        g = 100 * (k + np.sum(np.power(solutions[:, -k:] - 0.5, 2.0)-np.cos(20*np.pi*(solutions[:, -k:] - 0.5)), axis=1))
        f = [1.0 + g.copy() for _ in range(self.nobjs)]

        for i in range(self.nobjs):
            for j in range(self.nobjs - 1 - i):
                f[i] *= np.cos(0.5 * np.pi * solutions[:, j])
            if i > 0:
                f[i] *= np.sin(0.5 * np.pi * solutions[:, self.nobjs - i - 1])
        return f[i]


class OFAR(Problem):
    def __init__(self, nvars, nobjs, bounds, data_x, data_y, need_repair=False):
        super(OFAR, self).__init__(nvars, nobjs, bounds, need_repair)
        self.data_x = data_x
        self.data_y = data_y

    def evaluate_all(self, solutions):
        n_sol = solutions.shape[0]
        n_coef = self.data_x.shape[1]
        dim2 = self.data_x.shape[2]
        coef = solutions.reshape((n_sol, n_coef, dim2))
        yp = np.array([np.sum(self.data_x * coef[i], axis=1) for i in range(n_sol)])

        Q = -np.average(np.array([[self.include(x, y) for x, y in zip(self.data_y, yp[i])] for i in range(n_sol)]),
                        axis=1)
        V = np.zeros(n_sol)
        return np.stack([Q, V], axis=1)

    def evaluate_one(self, solutions, i):
        n_sol = solutions.shape[0]
        n_coef = self.data_x.shape[1]
        dim2 = self.data_x.shape[2]
        coef = solutions.reshape((n_sol, n_coef, dim2))
        yp = np.array([np.sum(self.data_x * coef[i], axis=1) for i in range(n_sol)])
        if i == 0:
            return -np.average(np.array([[self.include(x, y) for x, y in zip(yp[i], self.data_y)]
                                         for i in range(n_sol)]), axis=1)
        if i == 1:
            return np.zeros(n_sol)

    @staticmethod
    def include(x, y):
        x_ = x.reshape((2, -1))
        y_ = y.reshape((2, -1))
        # order
        zx = x_[0] <= x_[1]
        zy = y_[0] <= y_[1]
        if not np.all(zx == zy):
            return 0
        # coverage
        f1 = np.min([y_[0], y_[1]], axis=0) <= x_[0]
        f2 = x_[0] <= np.max([y_[0], y_[1]], axis=0)
        g1 = np.min([y_[0], y_[1]], axis=0) <= x_[1]
        g2 = x_[1] <= np.max([y_[0], y_[1]], axis=0)
        r = np.all([np.all([f1, f2], axis=0), np.all([g1, g2], axis=0)], axis=0).astype(int)
        return np.sum(r*np.linspace(1, 2, r.shape[0]))



