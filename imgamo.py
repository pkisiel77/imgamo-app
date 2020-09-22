import time
import numpy as np
import copy
import multiprocessing as mp
from utils import assigning_gens, get_not_dominated, front_suppression
from scipy.spatial.distance import cdist


class Options(object):
    def __init__(self, **kwargs):
        self.population_size = kwargs['population_size'] if 'population_size' in kwargs else 100
        self.max_evaluations = kwargs['max_evaluations'] if 'max_evaluations' in kwargs else -1
        self.max_iterations = kwargs['max_iterations'] if 'max_iterations' in kwargs else 1000
        self.exchange_iter = kwargs['exchange_iter'] if 'exchange_iter' in kwargs else 3
        self.change_iter = kwargs['change_iter'] if 'change_iter' in kwargs else 3
        self.front_max = kwargs['front_max'] if 'front_max' in kwargs else 100
        self.seed = kwargs['seed'] if 'seed' in kwargs else None
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else 0


class IMGAMO(object):
    def __init__(self, problem, players, options, ref_front=None):
        st = time.process_time()
        npop = options.population_size
        nvars = problem.nvars
        nobjs = problem.nobjs
        bounds = problem.bounds
        if options.seed is None:
            np.random.seed()
        else:
            np.random.seed(options.seed)
        self.options = options
        self.problem = problem
        self.patterns = assigning_gens(nvars, nobjs)
        self.players = players
        self.result = Result(options.front_max, nobjs, ref_front=ref_front)
        with mp.Pool(processes=problem.nobjs) as pool:
            pops = pool.starmap(worker_create, zip(players, (npop,)*nobjs, (nvars,)*nobjs, (bounds,)*nobjs))
            if problem.need_repair:
                pops = pool.starmap(worker_repair, zip((problem,)*nobjs, pops))
            pops_eval = pool.starmap(worker_evaluate, zip((problem,)*nobjs, pops))
        self.populations = np.stack(pops)
        self.populations_eval = np.stack(pops_eval)
        self.result.evaluation_count = np.zeros(nobjs) + np.size(self.populations_eval)
        self.result.elapsed_time += time.process_time() - st

    def run_algorithm(self):
        st = time.process_time()
        pool = mp.Pool(processes=self.problem.nobjs)
        iteration = 0
        # Main loop
        while iteration < self.options.max_iterations:
            # incrementation
            iteration += 1
            # optimize
            res = pool.starmap(worker_optimize, zip(self.players, self.populations, self.populations_eval,
                                                    self.patterns, (self.problem,)*self.problem.nobjs))
            for i in range(len(res)):
                self.result.evaluation_count += res[i][2]
            self.populations = np.stack([res[i][0] for i in range(len(res))])
            self.populations_eval = np.stack([res[i][1] for i in range(len(res))])
            self.players = [res[i][3] for i in range(len(res))]
            # get not dominated
            res = pool.map(get_not_dominated, self.populations_eval)
            not_dominated = np.vstack([pop[r] for pop, r in zip(self.populations, res)])
            not_dominated_eval = np.vstack([pop[r] for pop, r in zip(self.populations_eval, res)])
            mask = get_not_dominated(not_dominated_eval)
            not_dominated = not_dominated[mask]
            not_dominated_eval = not_dominated_eval[mask]
            self.result.add_to_front(not_dominated, not_dominated_eval)

            # exchange gens
            if iteration % self.options.exchange_iter == 0 and iteration < self.options.max_iterations:
                best_ind = [np.argmin(self.populations_eval[i, :, i]) for i in range(self.populations_eval.shape[0])]
                for i in range(self.populations.shape[0]):
                    for j in range(self.populations.shape[0]):
                        if i != j:
                            ind = (np.tile(np.arange(self.populations.shape[1]), len(np.where(self.patterns[j])[0])),
                                   np.repeat(np.where(self.patterns[j])[0], self.populations.shape[1]))
                            self.populations[i][ind] = np.repeat(self.populations[j, best_ind[j], self.patterns[j]],
                                                                 self.populations.shape[1])
                if self.problem.need_repair:
                    self.populations = np.stack(pool.starmap(worker_repair, zip((self.problem,) * self.problem.nobjs,
                                                                                self.populations)))
                self.populations_eval = np.stack(pool.starmap(worker_evaluate, zip((self.problem,)*self.problem.nobjs,
                                                                                   self.populations)))
                self.result.evaluation_count += np.size(self.populations_eval)

            # change patterns
            if iteration % self.options.change_iter == 0 and iteration < self.options.max_iterations:
                self.patterns = assigning_gens(self.problem.nvars, self.problem.nobjs)

            # verbose
            if self.options.verbose > 0 and iteration % self.options.verbose == 0:
                print('Iteration: ', iteration)
                print('Evaluation count: ', self.result.evaluation_count)
                print('Front size: ', self.result.front_size)
                print('Elapsed time:', time.process_time() - st)
                print('Metrics', self.result.metrics)
                print('')

            # stop condition
            if np.max(self.result.evaluation_count) > self.options.max_evaluations > -1:
                self.result.iteration = iteration
                break

        # verbose
        if self.options.verbose >= 0:
            print('Iteration: ', iteration)
            print('Evaluation count: ', self.result.evaluation_count)
            print('Front size: ', self.result.front_size)
            print('Elapsed time:', time.process_time() - st)
            print('Metrics', self.result.metrics)

        pool.close()
        pool.join()
        self.result.iteration = iteration
        self.result.elapsed_time += time.process_time() - st


class Result(object):
    def __init__(self, front_max, nobjs, ref_front=None):
        self.front_max = front_max
        self.front = []
        self.evaluated_front = []
        self.front_size = len(self.front)
        self.evaluation_count = np.zeros(nobjs)
        self.elapsed_time = 0
        self.iteration = 0
        self.ref_front = ref_front
        self.metrics = {'gd': None, 'igd': None}

    def add_to_front(self, not_dominated, not_dominated_eval):
        if self.front_size == 0:
            self.front = copy.deepcopy(not_dominated)
            self.evaluated_front = copy.deepcopy(not_dominated_eval)
            self.front_size = len(self.front)
        else:
            front = np.vstack([self.front, not_dominated])
            front_eval = np.vstack([self.evaluated_front, not_dominated_eval])
            mask = get_not_dominated(front_eval)
            self.front = front[mask]
            self.evaluated_front = front_eval[mask]
            self.front_size = len(self.front)
        # front suppression
        if self.front_size > self.front_max:
            mask = front_suppression(self.evaluated_front, self.front_max)
            self.front = self.front[mask]
            self.evaluated_front = self.evaluated_front[mask]
            self.front_size = len(self.front)
        if self.ref_front is not None:
            self.metrics['gd'] = np.mean(np.min(cdist(self.evaluated_front, self.ref_front), axis=1))
            self.metrics['igd'] = _igd(self.evaluated_front, self.ref_front)

    def summary(self):
        print('Iterations: ', self.iteration)
        print('Evaluation count: ', self.evaluation_count)
        print('Evaluation count per iteration: ', self.evaluation_count/self.iteration)
        print('Front size: ', self.front_size)
        print('Time:', self.elapsed_time)
        print('Time per iteration:', self.elapsed_time/self.iteration)
        print('Metrics', self.metrics)

    def plot_2d(self, ax, func_no1, func_no2):
        ax.scatter(self.evaluated_front[:, func_no1], self.evaluated_front[:, func_no2], marker='o', label='IMGAMO')
        ax.grid(True)
        ax.legend()

    def plot_3d(self, ax, func_no1, func_no2, func_no3):
        ax.scatter(self.evaluated_front[:, func_no1], self.evaluated_front[:, func_no2], self.evaluated_front[:, func_no3], marker='o', label='IMGAMO')
        ax.grid(True)
        ax.legend()


def worker_optimize(player, solutions, solutions_eval, pattern, problem):
    return player.optimize(solutions, solutions_eval, pattern, problem)


def worker_create(player, npop, nvars, bounds):
    return player.create_population(npop, nvars, bounds)


def worker_evaluate(problem, solutions):
    return problem.evaluate_all(solutions)


def worker_repair(problem, solutions):
    return problem.repair(solutions)


def vectorized_cdist(A, B, func_dist):
    u = np.repeat(A, B.shape[0], axis=0)
    v = np.tile(B, (A.shape[0], 1))

    D = func_dist(u, v)
    M = np.reshape(D, (A.shape[0], B.shape[0]))
    return M


def _igd(fpred, ftrue):
    N = np.max(ftrue, axis=0) - np.min(ftrue, axis=0)

    def dist(A, B):
        return np.sqrt(np.sum(np.square((A - B) / N), axis=1))
    D = vectorized_cdist(ftrue, fpred, dist)
    return np.mean(np.min(D, axis=1))

