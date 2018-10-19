import sympy as sp

x = sp.Symbol('x')

x_function = x * x * x
x_der1 = sp.diff(x_function, x, 1)
x_der2 = sp.diff(x_function, x, 2)

sp.plot(x_function, x_der1,x_der2, (x, -1, 1))

print(x_der1)
