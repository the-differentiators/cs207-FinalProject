import numpy as np

class Ad_Var():
    def __init__(self, val, ders=1):
        self._val = val
        self._ders = ders

    def __eq__(self, other):
        return self._val == other._val and self._ders == other._ders

    def __repr__(self):
        if type(self.get_ders()).__name__ == 'ndarray' and len(self.get_ders()) > 1:
            print_stm = f'Value = {self._val}\nGradient = {self._ders}'
        else:
            print_stm = f'Value = {self._val}\nDerivative = {self._ders}'
        return print_stm

    def get_val(self):
        return self._val

    def get_ders(self):
        return self._ders

    def __add__(self, other):
        try:
            return Ad_Var(self._val + other._val, self._ders + other._ders)
        except AttributeError:
            return Ad_Var(self._val + other, self._ders)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            return Ad_Var(self._val * other._val, self._ders * other._val + self._val * other._ders)
        except AttributeError:
            return Ad_Var(other * self._val, other * self._ders)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        pass

    def __rdiv__(self, other):
        pass

    def __abs__(self):
        pass

    def __pow__(self, r):
        if not (isinstance(r, float) or isinstance(r, int)):
            raise TypeError("Exponent must be of numeric type.")
        return Ad_Var(self._val ** r, r * self._val ** (r - 1) * self._ders)

    def exp(self):
        return Ad_Var(np.exp(self._val), np.exp(self._val) * self._ders)

    def log(self, logbase=np.e):
        pass

    def sin(self):
        pass

    def cos(self):
        pass

    def tan(self):
        pass

    def arcsin(self):
        pass

    def arccos(self):
        pass

    def arctan(self):
        pass

    @staticmethod
    def get_jacobian(functions_array, functions_dim, vars_dim):
        #input is a numpy array of Ad_Var function
        jacobian = np.zeros((functions_dim, vars_dim))
        for i, function in enumerate(functions_array):
            if type(function).__name__ != 'Ad_Var':
                raise TypeError("The list of functions inputted is not a numpy array of Ad_Var objects.")
            if (function.get_ders().shape[0] != vars_dim):
                raise ValueError(f"A function has variables defined in space with dimensions other than R^{vars_dim}")
            jacobian[i] = function.get_ders()
        return jacobian

    @staticmethod
    def get_values(functions_array):
        values = []
        for function in functions_array:
            if type(function).__name__ != 'Ad_Var':
                raise TypeError("The list of functions inputted is not a numpy array of Ad_Var objects.")
            values.append(function.get_val())
        return np.array(values)

if __name__ == "__main__":
    x = Ad_Var(2, np.array([1, 0]))
    y = Ad_Var(3, np.array([0, 1]))
    f = 3*x + 2*y + x*y
    print("Gradient example")
    print(f)
    z = Ad_Var(3)
    g = 3*z*z + 5
    print("scalar example")
    print(g)
    h = x*y
    g = np.array([f, h])
    print("jacobian example 1")
    print(Ad_Var.get_jacobian(g, 2, 2))
    x = Ad_Var(1, np.array([1, 0, 0]))
    y = Ad_Var(2, np.array([0, 1, 0]))
    z = Ad_Var(3, np.array([0, 0, 1]))
    f = np.array([x, y*x, z**2, 2*x+y+z, Ad_Var.exp(2*x)])
    print("jacobian example 2")
    print(Ad_Var.get_jacobian(f, 5, 3))
    print("values of vector function")
    print(Ad_Var.get_values(f))

