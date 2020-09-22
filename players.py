import numpy as np
from utils import population_suppression
from operators import create_individual_uniform, hiper_mutate
import copy
from scipy.optimize import minimize, differential_evolution


# base class for player
class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id

    def create_individual(self, nvars, bounds):
        return create_individual_uniform(nvars, bounds)

    def create_population(self, npop, nvars, bounds):
        pop = np.zeros((npop, nvars))
        for i in range(npop):
            pop[i] = self.create_individual(nvars, bounds)
        return pop

    def optimize(self, solutions, solutions_eval, pattern, bounds):
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = solutions.copy()
        new_solutions_eval = solutions_eval.copy()
        return new_solutions, new_solutions_eval, evaluation_count, self


class ClonalSelection(Player):
    def __init__(self, player_id, nclone=15, supp_level=0, mutate=hiper_mutate, mutate_args=(0.45, 0.9, 0.1)):
        super(ClonalSelection, self).__init__(player_id)
        self.nclone = nclone
        self.supp_level = supp_level
        self.mutate = mutate
        self.mutare_args = mutate_args

    def optimize(self, solutions, solutions_eval, pattern, problem):
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = copy.deepcopy(solutions)
        new_solutions_eval = copy.deepcopy(solutions_eval)
        temp_pop_eval = copy.deepcopy(solutions_eval[:, self.player_id])
        arg_sort = temp_pop_eval.argsort()
        indices = []
        better = []
        clone_num = self.nclone
        for arg in arg_sort:
            clones = np.array([self.mutate(copy.deepcopy(solutions[arg]), pattern, problem.bounds, *self.mutare_args)
                               for _ in range(clone_num)])
            if problem.need_repair:
                clones = problem.repair(clones)
            clones_eval = problem.evaluate_one(clones, self.player_id)
            evaluation_count[self.player_id] += clone_num
            argmin = clones_eval.argmin()
            if clones_eval[argmin] < solutions_eval[arg, self.player_id]:
                indices.append(arg)
                better.append(clones[argmin])
            clone_num = clone_num - 1 if clone_num > 2 else 1
        if len(better) > 0:
            better = np.stack(better)
            better_eval = problem.evaluate_all(better)
            evaluation_count += np.size(better_eval)
            new_solutions[indices] = better
            new_solutions_eval[indices] = better_eval
        # suppression
        if self.supp_level > 0:
            mask = population_suppression(new_solutions, self.supp_level)
            new = self.create_population(np.sum(mask), problem.nvars, problem.bounds)
            if problem.need_repair:
                new = problem.repair(new)
            new_eval = problem.evaluate_all(new)
            evaluation_count += np.size(new_eval)
            new_solutions[mask] = new
            new_solutions_eval[mask] = new_eval
        return new_solutions, new_solutions_eval, evaluation_count, self


class SimulatedAnnealing(Player):
    def __init__(self, player_id, temp=500, dec_step=0.99, mutate=hiper_mutate, mutate_args=(0.45, 0.9, 0.1)):
        super(SimulatedAnnealing, self).__init__(player_id)
        self.temp = temp
        self.dec_step = dec_step
        self.mutate = mutate
        self.mutare_args = mutate_args

    def optimize(self, solutions, solutions_eval, pattern, problem):
        npop = solutions.shape[0]
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = copy.deepcopy(solutions)
        new_solutions_eval = copy.deepcopy(solutions_eval)
        args = (pattern, problem.bounds) + self.mutare_args

        clones = np.apply_along_axis(self.mutate, 1, new_solutions, *args)
        if problem.need_repair:
            clones = problem.repair(clones)
        clones_eval = problem.evaluate_one(clones, self.player_id)
        evaluation_count[self.player_id] += npop
        r = np.random.random(size=npop)
        p = np.exp((new_solutions_eval[:, self.player_id] - clones_eval) / self.temp)
        mask = r < p
        new_solutions[mask, :] = clones[mask, :]
        new_solutions_eval[mask, :] = problem.evaluate_all(new_solutions[mask, :])
        evaluation_count += len(mask)
        self.temp = self.temp * self.dec_step
        return new_solutions, new_solutions_eval, evaluation_count, self


class SimpleGeneticAlg(Player):
    def __init__(self, player_id, pc=0.8, pm=0.05, dx=0.001):
        super(SimpleGeneticAlg, self).__init__(player_id)
        self.pc = pc
        self.pm = pm
        self.dx = dx

    def optimize(self, solutions, solutions_eval, pattern, problem):
        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = copy.deepcopy(solutions)
        new_solutions_eval = copy.deepcopy(solutions_eval)
        # bits, ndxs
        a = np.array([bnd[0] for bnd in problem.bounds])
        b = np.array([bnd[1] for bnd in problem.bounds])
        bits, ndxs = self._nbits(self.dx, a, b)
        # encode population
        pop = self._encode(solutions, bits, ndxs, a)
        # rulette
        pop_eval = -new_solutions_eval[:, self.player_id]
        minv = np.min(pop_eval)
        if minv <= 0:
            temp = pop_eval + np.abs(minv) + 1
        else:
            temp = pop_eval[:]
        mask = np.random.choice(pop.shape[0], pop.shape[0], replace=True, p=temp / np.sum(temp))
        pop = pop[mask, :]
        # cross
        for i in range(0, pop.shape[0], 2):
            if i + 1 >= pop.shape[0]:
                break
            r = np.random.random()
            if r < self.pc:
                nb = np.random.randint(1, len(pop[i]))
                child1 = pop[i, :].copy()
                child2 = pop[i+1, :].copy()
                temp = child1[:].copy()
                child1[nb:] = child2[nb:]
                child2[nb:] = temp[nb:]
                pop[i, np.repeat(pattern, bits)] = child1[np.repeat(pattern, bits)]
                pop[i + 1, np.repeat(pattern, bits)] = child2[np.repeat(pattern, bits)]
        # mutate
        mask = np.tile(np.repeat(pattern, bits), (pop.shape[0], 1))
        mask = np.logical_and(mask, np.random.random(size=pop.shape) < self.pm)
        pop = pop ^ mask
        # decode
        new_solutions = self._decode(pop, bits, ndxs, a)
        new_solutions_eval = problem.evaluate_all(new_solutions)
        evaluation_count += len(new_solutions)
        return new_solutions, new_solutions_eval, evaluation_count, self

    @staticmethod
    def _nbits(dx, a, b):
        lens = np.abs(b - a)
        lens_int = (np.ceil(lens / dx))
        bits = np.array([int(a).bit_length() for a in lens_int], dtype=int)
        ndxs = lens / (2 ** bits - 1)
        return bits, ndxs

    @staticmethod
    def _encode(solutions, bits, ndxs, a):
        def _tobinary(arr, bits):
            binary = np.array([], dtype=int)
            for x, b in zip(arr, bits):
                binary = np.append(binary, np.array(list(np.binary_repr(x, width=b)), dtype=int))
            return np.stack(binary)

        shape = solutions.shape
        pop = ((solutions - a) / ndxs).astype(int)
        pop = np.apply_along_axis(_tobinary, 1, pop, bits)
        return pop

    @staticmethod
    def _decode(population, bits, ndxs, a):
        def _decode_ind(individual, bits, ndxs, a):
            decode_individual = np.array([])
            s = 0
            for i in range(len(bits)):
                temp = individual[s:s + bits[i]]
                s += bits[i]
                decode = np.sum(temp * (np.ones(bits[i]) * 2) ** np.arange(bits[i] - 1, -1, -1)) * ndxs[i] + a[i]
                decode_individual = np.append(decode_individual, decode)
            return decode_individual
        return np.apply_along_axis(_decode_ind, 1, population, bits, ndxs, a)


class ScipyMinimize(Player):
    def __init__(self, player_id, method='L-BFGS-B', niter=1, strategy='best1bin'):
        # method: L-BFGS-B, SLSQP, TNC, differential_evolution
        super(ScipyMinimize, self).__init__(player_id)
        self.method = method
        self.options = {'maxiter': niter, 'disp': False}
        self.niter = niter
        self.strategy = strategy

    def optimize(self, solutions, solutions_eval, pattern, problem):
        def func(x, y):
            xx = np.zeros((1, pattern.shape[0]))
            xx[0, pattern] = x[:]
            xx[0, np.logical_not(pattern)] = y[:]
            return problem.evaluate_one(xx, self.player_id)[0]

        evaluation_count = np.zeros(solutions_eval.shape[1])
        new_solutions = copy.deepcopy(solutions)

        bnds = []
        for i in range(len(problem.bounds)):
            if pattern[i]:
                bnds.append(problem.bounds[i])

        for i in range(new_solutions.shape[0]):
            x = new_solutions[i, pattern].copy()
            y = new_solutions[i, np.logical_not(pattern)].copy()
            if self.method == 'differential_evolution':
                res = differential_evolution(func, tuple(bnds), args=(y,), strategy=self.strategy, maxiter=self.niter,
                                             init=new_solutions[:, pattern])
            else:
                res = minimize(func, x, (y,), method=self.method, bounds=tuple(bnds), options=self.options)
            new_solutions[i, pattern] = res.x.copy()
            new_solutions[i, np.logical_not(pattern)] = y.copy()
            evaluation_count[self.player_id] += res.nfev

        new_solutions_eval = problem.evaluate_all(new_solutions)
        evaluation_count += new_solutions.shape[0]

        return new_solutions, new_solutions_eval, evaluation_count, self
