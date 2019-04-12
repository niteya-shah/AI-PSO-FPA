from functions import dist_from_zero, rastrigin, booth, Matyas, Rosenbrock, THC, McCormic
import numpy as np
from SPOM import PSOM
from SPO import PSO
from FPA import FPA

functions = [dist_from_zero, rastrigin, booth, Matyas, Rosenbrock, THC, McCormic]

for i in functions:
    print(i.__name__)
    #
    print("  PSOM")
    psom = PSOM(50, 0.03,  i(100),0.1, 0.1, 1.5)
    print(" ",psom.cost(psom.min_cost), end=" ")
    psom.optimize(100)
    print("   ",psom.cost(psom.min_cost))

    pso = PSO(50, 0.03, 0.1, 0.1, i(100))
    print("  PSO")
    print(" ",pso.cost(pso.min_cost), end=" ")
    pso.optimize(100)
    print("   ",pso.cost(pso.min_cost))

    print("  FPA")
    fpa = FPA(50, 0.8, 1.5, i(100))
    print(" ",fpa.cost(fpa.min_cost), end=" ")
    fpa.optimize(100)
    print("   ",fpa.cost(fpa.min_cost))
