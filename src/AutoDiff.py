import numpy as np
import numbers as numbers

class Ad_Var():
    def __init__(self, val=1, ders=1):
        if isinstance(val, numbers.Number):
            if isinstance(ders, numbers.Number):
                pass
            elif isinstance(ders, np.ndarray):
                if np.array(list(map(lambda x: isinstance(x, numbers.Number), ders))).all():
                    pass
                else:
                    raise TypeError("Seed vector should be a numpy array of numeric type.")
            else:
                raise TypeError("Seed vector should be a numpy array of numeric type.")
        else:
            raise TypeError("Value should be one instance of numeric type.")
        self._val = val
        self._ders = ders


    def get_val(self):
        return self._val

    def get_ders(self):
        return self._ders

    def __eq__(self, other):
        return self._val == other._val and self._ders == other._ders

    def __ne__(self, other):
        return self._val != other._val or self._ders != other._ders

    def __repr__(self):
        if type(self.get_ders()).__name__ == 'ndarray' and len(self.get_ders()) > 1:
            print_stm = f'Value = {self._val}\nGradient = {self._ders}'
        else:
            print_stm = f'Value = {self._val}\nDerivative = {self._ders}'
        return print_stm

    def __neg__(self):
        pass
    
    def __add__(self, other):
        try:
            return Ad_Var(self._val + other._val, self._ders + other._ders)
        except AttributeError:
            return Ad_Var(self._val + other, self._ders)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        try:
            return Ad_Var(self._val - other._val, self._ders - other._ders)
        except AttributeError:
            return Ad_Var(self._val - other, self._ders)

    def __rsub__(self, other):
        try:
            return Ad_Var(other._val - self._val, other._ders - self._ders)
        except AttributeError:
            return Ad_Var(other - self._val, self._ders)
    

    def __mul__(self, other):
        try:
            return Ad_Var(self._val * other._val, self._ders * other._val + self._val * other._ders)
        except AttributeError:
            return Ad_Var(other * self._val, other * self._ders)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try: # Ad_Var(3)/Ad_Var(4)
            return Ad_Var(self._val / other._val, ((self._ders * other._val) - (self._val * other._ders)) / (other._val ** 2))
        except AttributeError: # Ad_Var(3)/4
            other = Ad_Var(other,0)
            return Ad_Var(self._val / other._val, ((self._ders * other._val) - (self._val * other._ders)) / (other._val ** 2))

    def __rtruediv__(self, other):
        # 3/Ad_Var(4)
        return Ad_Var(other / self._val, - self._ders*other / (self._val) ** 2)
        '''except AttributeError:
            return Ad_Var(other / self._val , - other / (self._ders)**2)'''

    def __pow__(self, other):
        try:
            return Ad_Var(self._val ** other._val, self._val ** other._val * (other._ders * np.log(self._val) + other._val * self._ders/self._val))
        except AttributeError:
            return Ad_Var(self._val ** other, other * self._val ** (other - 1) * self._ders)

    def __rpow__(self, other):
        if isinstance(other, numbers.Number):
            return Ad_Var(other ** self._val, np.log(other) * (other ** self._val) * self._ders)
        else:
            raise TypeError("Base should be an instance of numeric type.")

    def sqrt(self):
        return self ** 0.5

    def exp(self):
        return Ad_Var(np.exp(self._val), np.exp(self._val) * self._ders)

    def log(self, logbase=np.e):
        return Ad_Var(np.log(self._val) / np.log(logbase), self._ders / (self._val * np.log(logbase)))  

    def sin(self):
        return Ad_Var(np.sin(self._val), self._ders*np.cos(self._val)) 

    def cos(self):
        return Ad_Var(np.cos(self._val), -self._ders*np.sin(self._val)) 

    def tan(self):
        return Ad_Var(np.tan(self._val), self._ders / np.cos(self._val) ** 2) 

    def arcsin(self):
        return Ad_Var(np.arcsin(self._val), self._ders / np.sqrt(1 - (self._val ** 2))) 

    def arccos(self):
        return Ad_Var(np.arccos(self._val), -self._ders / np.sqrt(1 - (self._val ** 2))) 

    def arctan(self):
        return Ad_Var(np.arctan(self._val), self._ders / (1 + self._val ** 2))

    def logistic(self):
        pass

    def sinh(self):
        pass

    def cosh(self):
        pass

    def tanh(self):
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

    @staticmethod
    def grid_eval(f, grid):
        pass


