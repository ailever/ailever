## [Mathematics] | [numpy](https://numpy.org/doc/stable/contents.html) | [github](https://github.com/numpy/numpy)

### Ordinary Differential Equation
```python
#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy

#%%
# equation
t, k, T_0, T_a = sympy.symbols("t, k, T_0, T_a")
T = sympy.Function("T")
ode = T(t).diff(t) + k*(T(t) - T_a)

#sympy.Eq(ode)          # equation
#sympy.dsolve(ode)      # solution
#sympy.dsolve(ode).lhs  # left-hand-side
#sympy.dsolve(ode).rhs  # right-hand-side
init = {T(0): T_0}      # initial conditions
cons_eq = sympy.dsolve(ode).subs(t, 0).subs(T(0), T_0)
cons = sympy.solve(cons_eq)
solution = sympy.dsolve(ode).subs(cons[0]).rhs


f = sympy.lambdify((t, k, T_0, T_a), solution)
t = np.linspace(-1, 10, 300)
k = [0.1, 0.3, 0.5]
T_0 = [0.2, 0.3, 1]
T_a = [0.1]

plt.figure(figsize=(12,10))
for k_ in k:
    for T_0_ in T_0:
        for T_a_ in T_a:
            plt.plot(t, f(t, k_ ,T_0_ , T_a_), label=f'$k$={k_}, $t_0$={T_0_}, $t_a$={T_a_}')


plt.axhline(0, c='black', lw=0.7)
plt.axvline(0, c='black', lw=0.7)
plt.grid(True)
plt.legend()
```
![image](https://user-images.githubusercontent.com/52376448/99898872-5aa91f80-2ce8-11eb-86c1-74bace6b31b4.png)
