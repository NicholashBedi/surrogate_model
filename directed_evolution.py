import numpy as np
from scipy import optimize
from objective_functions import ObjectiveFunction

obf = ObjectiveFunction("easoms")
results = optimize.differential_evolution(obf.wrap_calc, obf.bounds,
                                    args = tuple([0]))
print(results)
obf.print_goal()
