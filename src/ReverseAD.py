import numpy as np
import numbers as numbers

class rAd_Var():
    def __init__(self, val):
        """
        Initialization of the rAd_Var class, with only the value of the variable passed in as an argument
        Parameters
        ==========
        self: Ad_Var
        val: variable value
        Examples
        =========
        >>> x = rAd_Var(3) #initializes an Ad_Var instance with value 3
        """
        if type(val) is np.ndarray and len(val) == 1:
            self._val = val[0]

        elif isinstance(val, numbers.Number):
            self._val = val

        else:
            self._val = np.array([val]).reshape(-1, )

        self._ders = None
        self.parents = []
        self.children = []
        self.seen = False # Set to True during runreverse() traversal, then reset at end

    def __str__(self):
        """
        Returns Returns a string representing the value of `self._val` (Value) and the value of `self._ders` (Gradient)
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        object representation for the Ad_Var instance
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> print(x)
        Value = 3
        Derivative = 1
        """
        print_stm = f'Value = {self._val}\nDerivative = {self.runreverse()}'
        return print_stm

    def __repr__(self):
        """
        Returns Returns a string representing the value of `self._val` (Value) and the value of `self._ders` (Gradient)
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        object representation for the Ad_Var instance
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> print(x)
        Value = 3
        Derivative = 1
        """
        print_stm = f'Value = {self._val}\nDerivative = {self.runreverse()}'
        return print_stm

    def set_val(self, value):
        """
        Sets the value for the rAd_Var instance passed.
        Parameters
        ==========
        self: rAd_Var
        value: variable value to be set
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> x.set_val(4)
        >>> x.get_val()
        4
        """
        self._val = value

    def set_ders(self, derivatives):
        """
        Sets the derivative for the rAd_Var instance passed.
        Parameters
        ==========
        self: rAd_Var
        derivatives: variable derivative to be set
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> x.set_ders(4)
        >>> x.get_ders()
        4
        """
        self._ders = derivatives

    def get_val(self):
        """
        Returns the value for the rAd_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        value for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> x.get_val()
        3
        """
        return self._val

    def get_ders(self):
        """
        Returns the derivative for the rAd_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        derivative for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> x.get_ders()
        1
        """
        if self._ders is None:
            # if self.children == []:
            #     self.set_ders(1)
            # else:
            new_deriv = sum(weight * var.get_ders() for var, weight in self.children)
            self.set_ders(new_deriv)
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
        """
        Returns whether the two rAd_Var instances passed have the same values and derivatives.
        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var
        Returns
        =======
        equality check for two rAd_Var instances
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> y = rAd_Var(3) * rAd_Var(1)
        >>> z = rAd_Var(4)
        >>> x == y
        True
        >>> x == z
        False
        """
        if self._val == other._val and self.get_ders() == other.get_ders():
            return True
        else:
            return False

    def __ne__(self, other):
        """
        Returns whether the two Ad_Var instances passed do not have the same values and derivatives.
        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var
        Returns
        =======
        inequality check for two Ad_Var instances
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> y = rAd_Var(3) * rAd_Var(1)
        >>> z = rAd_Var(4)
        >>> x != y
        False
        >>> x != z
        True
        """
        return not self == other

    def __neg__(self):
        """
        Returns the negation for the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        negation for the Ad_Var instance
        Examples
        =========
        >>> x = - rAd_Var(3)
        >>> x
        Value = -3
        Derivative = [-1]
        """
        rad_object = rAd_Var(self._val * -1)
        self.children.append((rad_object, -1.0))
        rad_object.parents = [self]
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
        rad_object.parents = [self]
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

    def get_originals(self):
        # Walk through tree, finding nodes without parents
        ancestorlist = []
        if self.parents:
            for parent in self.parents:
                if not parent.seen:
                    ancestorlist.append(parent) 
                    ancestorlist += parent.get_originals()
                    parent.seen = True
        
        # Reset all nodes in tree to unseen for future traversals
        for ancestor in ancestorlist:
            ancestor.seen = False
        
        return ancestorlist

    def runreverse(self):
        self.set_ders(1.0)
        originals = []
        seen_ids = []
        gradient_matrix = np.array([])
        for ancestor in self.get_originals():
            if not ancestor.parents and id(ancestor) not in seen_ids:
                originals.append(ancestor)
                seen_ids.append(id(ancestor))

        for original in originals:
            gradient_matrix = np.append(gradient_matrix, original.get_ders())

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
        ...  return x * y
        >>> def f2(x,y):
        ...  return x + y
        >>> def f3(x, y):
        ...  return rAd_Var.log(x ** y)
        >>> rAd_Var.get_jacobian([f1, f2, f3], [1, 2])
        array([[2., 1.],
               [1., 1.],
               [2., 0.]])
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

    @staticmethod
    def get_values(functions_array, var_values):
        """
        Returns the values of for a vector-valued function evaluated at a point in higher dimensions.
        INPUTS
        =======
        functions_array: numpy array of Python function
            a vector of functions passed into the method
        var_values: numpy array of numeric values
           values for variables in the functions array
        RETURNS
        ========
        values: a numpy array with shape (functions_dim,)
            the vector with the values of the vector-valued function evaluated at a point
        NOTES
        =====
        PRE:
             - functions_array is a numpy array of only Ad_Var objects
        POST:
             - raises a TypeError exception if any of the elements of the functions_array are not of type Ad_Var
        EXAMPLES
        =========
        >>> def f1(x, y):
        ...  return x * y
        >>> def f2(x,y):
        ...  return x + y
        >>> def f3(x, y):
        ...  return rAd_Var.log(x ** y)
        >>> rAd_Var.get_values([f1, f2, f3], [1, 2])
        array([2., 3., 0.])
        """
        values = []
        for function in functions_array:
            variables = []
            for value in var_values:
                variables.append(rAd_Var(value))
            if len(function.__code__.co_varnames) > len(variables):
                raise ValueError(f"Number of arguments required for function is greater than the number of input variables ({len(var_values)}).")
            values.append(function(*variables).get_val())
        return np.array(values)

if __name__=='__main__':
    import doctest
    doctest.testmod(verbose = False)
    print("Passed all doctests!")