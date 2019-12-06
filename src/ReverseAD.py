import numpy as np
import numbers as numbers

class rAd_Var():
    def __init__(self, val, ders=None):
        self._val = val
        self._ders = ders
        self.parents = []
        self.children = []
        self.seen = False # Set to True during runreverse() traversal, then reset at end

    def __str__(self):
        return f'Reverse Autodiff Object with value {self._val} and gradient {self.gradient()}'

    def get_val(self):
        return self._val

    def get_ders(self):
        return self._ders

    def __add__(self, other):
        try:
            rad_object = rAd_Var(self._val + other._val)
            self.children.append((rad_object, 1))
            other.children.append((rad_object, 1))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val + other)
            self.children.append((rad_object, 1))
            rad_object.parents = [self]
            return rad_object

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            rad_object = rAd_Var(self._val - other._val)
            self.children.append((rad_object, 1))
            other.children.append((rad_object, -1 ))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val - other)
            self.children.append((rad_object, 1))
            rad_object.parents = [self]
            return rad_object

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        try:
            rad_object = rAd_Var(self._val * other._val)
            self.children.append((rad_object, other._val))
            other.children.append((rad_object, self._val))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val * other)
            self.children.append((rad_object, other))
            rad_object.parents = [self]
            return rad_object

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            rad_object = rAd_Var(self._val / other._val)
            self.children.append((rad_object, 1 / other._val))
            other.children.append((rad_object, -self._val/(other._val**2)))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val / other)
            self.children.append((rad_object, 1 / other))
            rad_object.parents = [self]
            return rad_object

    def __rtruediv__(self, other):
        try:
            return other / self
        except AttributeError:
            rad_object = rAd_Var(other / self._val)
            self.children.append((rad_object, - other / (self._val**2)))
            rad_object.parents = [self]
            return rad_object

    def __pow__(self, other):
        try:
            rad_object = rAd_Var(self._val ** other._val)
            self.children.append((rad_object, other._val * self._val ** (other._val - 1)))
            other.children.append((rad_object, self._val**other._val*np.log(self._val)))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val ** other)
            self.children.append((rad_object, other * self._val ** (other - 1)))
            rad_object.parents = [self]
            return rad_object

    def __rpow__(self, other):
        if isinstance(other, numbers.Number):
            rad_object = rAd_Var(other ** self._val)
            self.children.append((rad_object, other ** self._val * np.log(other)))
            rad_object.parents = [self]
            return rad_object
        else:
            raise TypeError("Base should be an instance of numeric type.")

    def __eq__(self, other):
        if self._val == other._val and self.gradient() == other.gradient():
            return True
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        rad_object = rAd_Var(self._val * -1)
        self.children.append(rad_object, (np.array([-1.0]*len(self._val))))
        return rad_object

    def sqrt(self):
        rad_object = rAd_Var(self._val ** 0.5)
        self.children.append((rad_object, 0.5 * self._val ** (-0.5)))
        rad_object.parents = [self]
        return rad_object

    def exp(self):
        rad_object = rAd_Var(np.exp(self._val))
        self.children.append((rad_object, np.exp(self._val)))
        rad_object.parents = [self]
        return rad_object

    def log(self, logbase=np.e):
        rad_object = rAd_Var(np.log(self._val))
        self.children.append((rad_object, 1/(self._val*np.log(logbase))))
        rad_object.parents = [self]
        return rad_object

    def sin(self):
        rad_object = rAd_Var(np.sin(self._val))
        self.children.append((rad_object, (np.cos(self._val))))
        return rad_object

    def cos(self):
        rad_object = rAd_Var(np.cos(self._val))
        self.children.append((rad_object, -(np.sin(self._val))))
        rad_object.parents = [self]
        return rad_object

    def tan(self):
        rad_object = rAd_Var(np.tan(self._val))
        self.children.append((rad_object, 1/(np.cos(self._val))**2))
        rad_object.parents = [self]
        return rad_object

    def logistic(self):
        rad_object = rAd_Var(1/(1+np.exp(-self._val)))
        self.children.append((rad_object, (np.exp(-self._val)/((np.exp(-self._val)+1)**2))))
        rad_object.parents = [self]
        return rad_object

    def arcsin(self):
        if -1 <= self._val <= 1:
            rad_object = rAd_Var(np.arcsin(self._val))
            self.children.append((rad_object, 1/np.sqrt(1 - (self._val ** 2))))
            rad_object.parents = [self]
            return rad_object
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def arccos(self):
        if -1 <= self._val <= 1:
            rad_object = rAd_Var(np.arccos(self._val))
            self.children.append((rad_object, -1/np.sqrt(1 - (self._val ** 2))))
            rad_object.parents = [self]
            return rad_object
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def arctan(self):
        rad_object = rAd_Var(np.arctan(self._val))
        self.children.append((rad_object, 1/(1 + self._val ** 2)))
        rad_object.parents = [self]
        return rad_object

    def sinh(self):
        rad_object = rAd_Var(np.sinh(self._val))
        self.children.append((rad_object, np.cosh(self._val)))
        rad_object.parents = [self]
        return rad_object

    def sinh(self):
        rad_object = rAd_Var(np.sinh(self._val))
        self.children.append((rad_object, np.cosh(self._val)))
        rad_object.parents = [self]
        return rad_object

    def cosh(self):
        rad_object = rAd_Var(np.cosh(self._val))
        self.children.append((rad_object, np.sinh(self._val)))
        rad_object.parents = [self]
        return rad_object

    def tanh(self):
        rad_object = rAd_Var(np.tanh(self._val))
        self.children.append((rad_object, (1 - np.tanh(self._val)**2)))
        rad_object.parents = [self]
        return rad_object

    def gradient(self):
        if self._ders is None:
            self._ders = 0
            for var, weight in self.children:
                self._ders += weight * var.gradient()
        return self._ders

    def get_originals(self):
        # Walk through tree, finding nodes without parents
        ancestorlist = []
        seen = []
        if self.parents:
            for parent in self.parents:
                if not parent.seen:
                    ancestorlist.append(parent) 
                    ancestorlist += parent.get_originals()
                    # parent.seen = True
        
        # Reset all nodes in tree to unseen for future traversals
        for ancestor in ancestorlist:
            ancestor.seen = False
        
        return ancestorlist

    def runreverse(self):
        self._ders = 1.0
        originals = []
        seen_ids = []
        gradient_matrix = np.array([])
        for ancestor in self.get_originals():
            if not ancestor.parents and id(ancestor) not in seen_ids:
                originals.append(ancestor)
                seen_ids.append(id(ancestor))

        for original in originals:
            gradient_matrix = np.append(gradient_matrix, original.gradient())

        return gradient_matrix

    @staticmethod
    def get_jacobian(functions_array, var_values):
        """
        Returns the jacobian matrix for a vector of functions, with given values for variables in the function
        INPUTS
        =======
        functions_array: numpy array of Python function
            a vector of functions passed into the method
        var_values: numpy array of numeric values
           values for variables in the functions array
        RETURNS
        ========
        jacobian: a numpy array with shape (len(functions_array), len(var_values)), the jacobian matrix of the vector-valued function
        NOTES
        =====
        PRE:
             - functions_array is a numpy array of only Ad_Var objects
             - the gradient vector of each function in functions_array must have dimensions equal to vars_dim
               i.e. all functions in functions_array live in a space of equal dimensions.
        POST:
             - the values or the derivatives of the functions in functions_array are not changed
             - the result of get_jacobian is a numpy 2D array and not an Ad_Var object
             - raises a ValueError exception if the number of arguments required for a function is greater than the number of input variables
        EXAMPLES
        =========
        >>> def f1(x, y):
        ...  rAd_Var.cos(x) * (y + 2)
        >>> def f2(x,y):
        ...  return 1 + x ** 2 / (x * y * 3)
        >>> def f3(x, y):
        ...  return 3 * rAd_Var.log(x * 2) + rAd_Var.exp(x / y)
        >>> rAd_Var.get_jacobian([f1, f2, f3], [1, 2]
        [[-3.36588394  0.54030231]
        [ 0.16666667 -0.08333333]
        [ 3.82436064 -0.41218032]]
        """

        #input is a numpy array of Ad_Var function
        functions_dim = len(functions_array)
        vars_dim = len(var_values)

        jacobian = np.zeros((functions_dim, vars_dim))
        for i, function in enumerate(functions_array):
            variables = []
            for value in var_values:
                variables.append(rAd_Var(value))
            if len(function.__code__.co_varnames) > len(variables):
                raise ValueError(f"Number of arguments required for function is greater than the number of input variables ({vars_dim}).")
            jacobian[i] = function(*variables).runreverse()

        return jacobian