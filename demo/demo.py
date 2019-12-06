import sys
sys.path.append('../')

import numpy as np
from src.AutoDiff import Ad_Var


x = Ad_Var()
f_str = "x**2 + 2*x"
dict = Ad_Var.grid_eval(f_str, ['x'], [x], [[2, 5]])
print(dict)

x = Ad_Var(1, np.array([1, 0]))
y = Ad_Var(2, np.array([0, 1]))
f_string = "[Ad_Var.cos(x) * (y + 2), 1 + x ** 2 / (x * y * 3), 3 * Ad_Var.log(x * 2) + Ad_Var.exp(x / y)]"
dict = Ad_Var.grid_eval(f_string, ['x','y'], [x, y], [[1,2],[2,3]])
print(dict)
