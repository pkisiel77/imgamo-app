import numpy as np


def create_individual_uniform(nvars, bounds):
    ind = np.random.random(nvars)
    for k in range(nvars):
        if bounds[k][0] is None and bounds[k][1] is None:
            ind[k] = np.math.tan(np.math.pi * ind[k] - np.math.pi*0.5)
        elif bounds[k][0] is not None and bounds[k][1] is None:
            a = bounds[k][0]
            ind[k] = -np.math.log(ind[k]) + a
        elif bounds[k][0] is None and bounds[k][1] is not None:
            b = bounds[k][1]
            ind[k] = np.math.log(ind[k]) + b
        else:
            a = bounds[k][0]
            b = bounds[k][1]
            ind[k] = (b - a) * ind[k] + a
    return ind


def uniform_mutate(individual, pattern, bounds):
    indx = np.where(pattern)[0]
    k = np.random.choice(indx)
    individual[k] = np.random.random()
    if bounds[k][0] is None and bounds[k][1] is None:
        individual[k] = np.math.tan(np.math.pi * individual[k] - np.math.pi*0.5)
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        individual[k] = -np.math.log(individual[k]) + a
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        individual[k] = np.math.log(individual[k]) + b
    else:
        a = bounds[k][0]
        b = bounds[k][1]
        individual[k] = (b - a) * individual[k] + a
    return individual


def bound_mutate(individual, pattern, bounds):
    indx = np.where(pattern)[0]
    k = np.random.choice(indx)
    if bounds[k][0] is None and bounds[k][1] is None:
        a = -100.0
        b = 100.0
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        b = a + 100.0
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        a = b - 100.0
    else:
        a = bounds[k][0]
        b = bounds[k][1]
    r = np.random.random()
    if r < 0.5:
        r = np.random.random()
        individual[k] = a + (individual[k] - a) * r
    else:
        r = np.random.random()
        individual[k] = (b - individual[k]) * r + individual[k]
    return individual


def gaussian_mutate(individual, pattern, bounds, sigma):
    indx = np.where(pattern)[0]
    k = np.random.choice(indx)
    if bounds[k][0] is None and bounds[k][1] is None:
        a = -100.0
        b = 100.0
    elif bounds[k][0] is not None and bounds[k][1] is None:
        a = bounds[k][0]
        b = a + 100.0
    elif bounds[k][0] is None and bounds[k][1] is not None:
        b = bounds[k][1]
        a = b - 100.0
    else:
        a = bounds[k][0]
        b = bounds[k][1]
    ran = sigma * np.random.randn() + individual[k]
    if a < ran < b:
        individual[k] = ran
    return individual


def hiper_mutate(ind, pattern, bounds, a, b, sigma):
    r = np.random.random()
    if r < a:
        ind = uniform_mutate(ind, pattern, bounds)
    elif r < b:
        ind = gaussian_mutate(ind, pattern, bounds, sigma)
    else:
        ind = bound_mutate(ind, pattern, bounds)
    return ind