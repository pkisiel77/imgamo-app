import numpy as np


def assigning_gens(nvars, nobjs):
    while True:
        r = np.random.randint(0, nobjs, size=(nvars,))
        r2 = np.stack([r == i for i in range(nobjs)])
        if not np.any(np.all(r2, axis=1)) and not np.any(np.all(np.logical_not(r2), axis=1)):
            break
    return r2


def pairwise_dominance(x):
    z = x[:, np.newaxis] > x
    z = np.all(z, axis=2)
    xx = np.any(z, axis=1)
    return np.logical_not(xx)


def get_not_dominated(populations_eval):
    mask = pairwise_dominance(populations_eval)
    return mask


def pairwise_distance(x):
    return np.sum((x[:, np.newaxis] - x)**2, axis=2)


def front_suppression(front_eval, front_max):
    n = front_eval.shape[0] - front_max
    z = pairwise_distance(front_eval)
    t = np.tril(z) + np.triu(np.ones_like(z) * 1000000)
    arg = np.argsort(t, axis=None)
    indx = np.unravel_index(arg, t.shape)[0]
    u, i = np.unique(indx, return_index=True)
    mask = np.ones(front_eval.shape[0], dtype=bool)
    mask[indx[np.sort(i)][:n]] = False
    return mask


def population_suppression(population, level):
    z = pairwise_distance(population)
    t = np.tril(z) + np.triu(np.ones_like(z) * level)
    ind = np.where(t < level)[0]
    u, i = np.unique(ind, return_index=True)
    mask = np.zeros(population.shape[0], dtype=bool)
    mask[ind[np.sort(i)]] = True
    return mask
