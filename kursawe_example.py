from problems import Kursawe
from players import ClonalSelection
import imgamo
import matplotlib.pyplot as plt

nvars = 3
nobjs = 2 # liczba funkcji celu
bounds = ((-5.0, 5.0),) * nvars

problem = Kursawe(nvars=nvars, nobjs=nobjs, bounds=bounds)
players = [ClonalSelection(i, nclone=25) for i in range(nobjs)]
options = imgamo.Options(max_iterations=100, population_size=25, verbose=0)

algorithm = imgamo.IMGAMO(problem, players, options)
print(algorithm.result.elapsed_time)
algorithm.run_algorithm()
print(algorithm.result.elapsed_time)
print(algorithm.result.front_size)
print(algorithm.result.evaluation_count)

fig, ax = plt.subplots()
algorithm.result.plot_2d(ax, 0, 1)
plt.show()

