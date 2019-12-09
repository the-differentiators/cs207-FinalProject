import numpy as np
import numbers as numbers
from itertools import product

class Ad_Var():
    def __init__(self, val=1, ders=1):
        """
        Initialization of the Ad_Var class, the default value of the Ad_Var instance is 1, and the default
        derivative of the Ad_Var instance is 1

        Parameters
        ==========
        self: Ad_Var
        val: variable value
        ders: variable derivative

        Examples
        =========
        >>> x = Ad_Var(3,2) #initializes an Ad_Var instance with value 3, and derivative 2
        """
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

    def set_val(self, value):
        """
        Sets the value for the Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var
        value: variable value to be set

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> x.set_val(4)
        >>> x.get_val()
        4
        """
        self._val = value

    def set_ders(self, derivatives):
        """
        Sets the derivative for the Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var
        derivatives: variable derivative to be set

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> x.set_ders(4)
        >>> x.get_ders()
        4
        """
        self._ders = derivatives

    def get_val(self):
        """
        Returns the value for the Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        value for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> x.get_val()
        3
        """
        return self._val

    def get_ders(self):
        """
        Returns the derivative for the Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        derivative for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> x.get_ders()
        1
        """
        return self._ders

    def _typecheck_other(self, other):
        if type(other) == rAd_Var:
            raise TypeError("Ad_Var object cannot be used with rAd_Var objects!")

    def __eq__(self, other):
        """
        Returns whether the two Ad_Var instances passed have the same values and derivatives.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var

        Returns
        =======
        equality check for two Ad_Var instances

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> y = Ad_Var(3,1)
        >>> z = Ad_Var(4,1)
        >>> x == y
        True
        >>> x == z
        False
        """
        self._typecheck_other(other)
        if np.isscalar(self._ders):
            if np.isscalar(other._ders):
                return self._val == other._val and self._ders == other._ders
            else:
                raise TypeError('Can not compare a scaler Ad_Var and a vector Ad_Var')
        else:
            if np.isscalar(other._ders):
                raise TypeError('Can not compare a scaler Ad_Var and a vector Ad_Var')
            else:
                return (self._val == other._val) and (self._ders == other._ders).all()

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
        >>> x = Ad_Var(3)
        >>> y = Ad_Var(3,1)
        >>> z = Ad_Var(4,1)
        >>> x != y
        False
        >>> x != z
        True
        """
        self._typecheck_other(other)
        if np.isscalar(self._ders):
            if np.isscalar(other._ders):
                return self._val != other._val or self._ders != other._ders
            else:
                raise TypeError('Can not compare a scaler Ad_Var and a vector Ad_Var')
        else:
            if np.isscalar(other._ders):
                raise TypeError('Can not compare a scaler Ad_Var and a vector Ad_Var')
            else:
                return (self._val != other._val) or (self._ders != other._ders).any()



    def __repr__(self):
        """
        Returns Returns a string representing the value of `self._val` (Value) and the value of `self._ders` (Gradient)

        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        object representation for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(3)
        >>> print(x)
        Value = 3
        Derivative = 1
        """
        if type(self.get_ders()).__name__ == 'ndarray' and len(self.get_ders()) > 1:
            print_stm = f'Value = {self._val}\nGradient = {self._ders}'
        else:
            print_stm = f'Value = {self._val}\nDerivative = {self._ders}'
        return print_stm

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
        >>> x = Ad_Var(3)
        >>> -x
        Value = -3
        Derivative = -1
        """
        return Ad_Var(-self._val, -self._ders)

    def __add__(self, other):
        """
        Returns the addition for one Ad_Var instance and another Ad_Var or numeric type instance passed.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type

        Returns
        =======
        addition for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(1)
        >>> f = x + y
        >>> f.get_val()
        2
        >>> f.get_ders()
        3
        """
        self._typecheck_other(other)
        try:
            return Ad_Var(self._val + other._val, self._ders + other._ders)
        except AttributeError:
            return Ad_Var(self._val + other, self._ders)

    def __radd__(self, other):
        """
        Returns the addition for one Ad_Var or numeric type instance and an Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var/numeric type
        other: Ad_Var

        Returns
        =======
        addition for the Ad_Var instance

        Examples
        =========
        >>> x = 1
        >>> y = Ad_Var(2)
        >>> f = x + y
        >>> f.get_val()
        3
        >>> f.get_ders()
        1
        """
        self._typecheck_other(other)
        return self.__add__(other)

    def __sub__(self, other):
        """
        Returns the subtraction for one Ad_Var instance and another Ad_Var or numeric type instance passed.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type

        Returns
        =======
        subtraction for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(1)
        >>> f = x - y
        >>> f.get_val()
        0
        >>> f.get_ders()
        1
        """
        self._typecheck_other(other)
        try:
            return Ad_Var(self._val - other._val, self._ders - other._ders)
        except AttributeError:
            return Ad_Var(self._val - other, self._ders)

    def __rsub__(self, other):
        """
        Returns the subtraction for one Ad_Var or numeric type instance and another Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var/numeric type
        other: Ad_Var

        Returns
        =======
        subtraction for the Ad_Var instance

        Examples
        =========
        >>> x = 1
        >>> y = Ad_Var(1)
        >>> f = x - y
        >>> f.get_val()
        0
        >>> f.get_ders()
        -1
        """
        self._typecheck_other(other)
        try:
            return Ad_Var(other._val - self._val, other._ders - self._ders)
        except AttributeError:
            return Ad_Var(other - self._val, - self._ders) #self._ders


    def __mul__(self, other):
        """
        Returns the multiplication for one Ad_Var instance and another Ad_Var or numeric type instance passed.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type

        Returns
        =======
        multiplication for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(2)
        >>> f = x * y
        >>> f.get_val()
        2
        >>> f.get_ders()
        5
        """
        self._typecheck_other(other)
        try:
            return Ad_Var(self._val * other._val, self._ders * other._val + self._val * other._ders)
        except AttributeError:
            return Ad_Var(other * self._val, other * self._ders)

    def __rmul__(self, other):
        """
        Returns the multiplication for one Ad_Var or numeric type instance and another Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var/numeric type
        other: Ad_Var

        Returns
        =======
        multiplication for the Ad_Var instance

        Examples
        =========
        >>> x = 2
        >>> y = Ad_Var(2)
        >>> f = x * y
        >>> f.get_val()
        4
        >>> f.get_ders()
        2
        """
        self._typecheck_other(other)
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Returns the division for one Ad_Var instance and another Ad_Var or numeric type instance passed.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type

        Returns
        =======
        division for the Ad_Var instance

        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(2,1)
        >>> f = x / y
        >>> f.get_val()
        0.5
        >>> f.get_ders()
        0.75
        """
        self._typecheck_other(other)
        try: # Ad_Var(3)/Ad_Var(4)
            return Ad_Var(self._val / other._val, ((self._ders * other._val) - (self._val * other._ders)) / (other._val ** 2))
        except AttributeError: # Ad_Var(3)/4
            other = Ad_Var(other,0)
            return Ad_Var(self._val / other._val, ((self._ders * other._val) - (self._val * other._ders)) / (other._val ** 2))

    def __rtruediv__(self, other):
        """
        Returns the division for one Ad_Var or numeric type instance and another Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var/numeric type
        other: Ad_Var

        Returns
        =======
        division for the Ad_Var instance

        Examples
        =========
        >>> x = 1
        >>> y = Ad_Var(2,1)
        >>> f = x / y
        >>> f.get_val()
        0.5
        >>> f.get_ders()
        -0.25
        """
        self._typecheck_other(other)
        return Ad_Var(other / self._val, - self._ders*other / (self._val) ** 2)

    def __pow__(self, other):
        """
        Returns the power for one Ad_Var instance to the another Ad_Var or numeric type instance passed.

        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type

        Returns
        =======
        power for the Ad_Var instance, self ** other

        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(2,1)
        >>> f = x ** y
        >>> f.get_val()
        1
        >>> f.get_ders()
        4.0
        """
        self._typecheck_other(other)
        try:
            return Ad_Var(self._val ** other._val, self._val ** other._val * (other._ders * np.log(self._val) + other._val * self._ders/self._val))
        except AttributeError:
            return Ad_Var(self._val ** other, other * self._val ** (other - 1) * self._ders)

    def __rpow__(self, other):
        """
        Returns the power for one Ad_Var or numeric type instance to another Ad_Var instance passed.

        Parameters
        ==========
        self: Ad_Var/numeric type
        other: Ad_Var

        Returns
        =======
        power for the Ad_Var instance

        Examples
        =========
        >>> x = 2
        >>> y = Ad_Var(2,1)
        >>> f = x ** y
        >>> f.get_val()
        4
        >>> f.get_ders()
        2.772588722239781
        """
        self._typecheck_other(other)
        if isinstance(other, numbers.Number):
            return Ad_Var(other ** self._val, np.log(other) * (other ** self._val) * self._ders)
        else:
            raise TypeError("Base should be an instance of numeric type.")

    def sqrt(self):
        """
        Returns the square root of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the square root of the object passed

    	Examples
        =========
        >>> x = Ad_Var(16)
        >>> f = Ad_Var.sqrt(x)
        >>> f.get_val()
        4.0
        >>> f.get_ders()
        0.125
        """
        return self ** 0.5

    def exp(self):
        """
        Returns the exponential of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the exponential of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(1,np.array([1, 0]))
        >>> x2 = Ad_Var(2,np.array([0, 1]))
        >>> f = Ad_Var.exp(x1 + x2)
        >>> f.get_val()
        20.085536923187668
        >>> f.get_ders()
        array([20.08553692, 20.08553692])
        """
        return Ad_Var(np.exp(self._val), np.exp(self._val) * self._ders)

    def log(self, logbase=np.e):
        """
        Returns the logarithm of the Ad_Var instance passed with the base specified by the user
        Parameters
        ==========
        self: Ad_Var
        logbase: float, optional, default value is np.e
                 the base of the logarithm

        Returns
        =======
        Ad_Var object which is the logarithm with base logbase of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(2,np.array([1, 0]))
        >>> x2 = Ad_Var(3,np.array([0, 1]))
        >>> f = Ad_Var.log(x1 + 2*x2)
        >>> f.get_val()
        2.0794415416798357
        >>> f.get_ders()
        array([0.125, 0.25 ])
        """
        return Ad_Var(np.log(self._val) / np.log(logbase), self._ders / (self._val * np.log(logbase)))

    def sin(self):
        """
        Returns the sine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the sine of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(np.pi/4)
        >>> f = Ad_Var.sin(x1)
        >>> f.get_val()
        0.7071067811865476
        >>> f.get_ders()
        0.7071067811865476
        """
        return Ad_Var(np.sin(self._val), self._ders*np.cos(self._val))

    def cos(self):
        """
        Returns the cosine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the cosine of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(np.pi/4)
        >>> f = Ad_Var.cos(x1)
        >>> f.get_val()
        0.7071067811865476
        >>> f.get_ders()
        -0.7071067811865476
        """
        return Ad_Var(np.cos(self._val), -self._ders*np.sin(self._val))

    def tan(self):
        """
        Returns the tangent of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the tangent of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(0)
        >>> f = Ad_Var.tan(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        1.0
        """
        return Ad_Var(np.tan(self._val), self._ders / np.cos(self._val) ** 2)

    def arcsin(self):
        """
        Returns the arcsin of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the arcsin of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(0)
        >>> f = Ad_Var.arcsin(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        1.0
        """
        if -1 <= self._val <= 1:
            return Ad_Var(np.arcsin(self._val), self._ders / np.sqrt(1 - (self._val ** 2)))
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')


    def arccos(self):
        """
        Returns the arccos of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the arccos of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(0)
        >>> f = Ad_Var.arccos(x1)
        >>> f.get_val()
        1.5707963267948966
        >>> f.get_ders()
        -1.0
        """
        if -1 <= self._val <= 1:
            return Ad_Var(np.arccos(self._val), -self._ders / np.sqrt(1 - (self._val ** 2)))
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def arctan(self):
        """
        Returns the arctan of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the arctan of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(0)
        >>> f = Ad_Var.arctan(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        1.0
        """
        if -1 <= self._val <= 1:
            return Ad_Var(np.arctan(self._val), self._ders / (1 + self._val ** 2))
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def sinh(self):
        """
        Returns the hyperbolic sine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the hyperbolic sine of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(np.pi/4)
        >>> f = Ad_Var.sinh(x1)
        >>> f.get_val()
        0.8686709614860095
        >>> f.get_ders()
        1.324609089252006
        """
        return Ad_Var(np.sinh(self._val), self._ders*np.cosh(self._val))

    def cosh(self):
        """
        Returns the hyperbolic cosine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the hyperbolic cosine of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(np.pi/4)
        >>> f = Ad_Var.cosh(x1)
        >>> f.get_val()
        1.324609089252006
        >>> f.get_ders()
        0.8686709614860095
        """
        return Ad_Var(np.cosh(self._val), self._ders*np.sinh(self._val))

    def tanh(self):
        """
        Returns the hyperbolic tangent of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which is the hyperbolic tangent of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(np.pi/4)
        >>> f = Ad_Var.tanh(x1)
        >>> f.get_val()
        0.6557942026326724
        >>> f.get_ders()
        0.5699339637933774
        """
        return Ad_Var(np.tanh(self._val), self._ders*(1 - np.tanh(self._val)**2))

    def logistic(self):
        """
        Returns the logistic function evaluated at value of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var

        Returns
        =======
        Ad_Var object which has value equal to sigmoid of the value of the object passed
        Examples
        =========
        >>> x1 = Ad_Var(0)
        >>> f = Ad_Var.logistic(x1)
        >>> f.get_val()
        0.5
        >>> f.get_ders()
        0.25
        >>> x1 = Ad_Var(-1, np.array([1, 0]))
        >>> x2 = Ad_Var(2, np.array([0, 1]))
        >>> f = Ad_Var.logistic(x1*x2)
        >>> f.get_val()
        0.11920292202211755
        >>> f.get_ders()
        array([ 0.20998717, -0.10499359])
        """
        def sigmoid(x):
            return 1.0/(1 + np.exp(-x))

        def sigmoid_derivative(x):
            der = (1 - sigmoid(x)) * sigmoid(x)
            return der

        return Ad_Var(sigmoid(self._val), sigmoid_derivative(self._val) * self._ders)


    @staticmethod
    def get_jacobian(functions_array, functions_dim, vars_dim):
        """
        Returns the jacobian matrix for a vector-valued function

        INPUTS
        =======
        functions_array: numpy array of Ad_Var objects
            a vector of functions which make up the vector-valued function together
        functions_dim: int
           the number of different functions which make up the vector-valued function
        vars_dim: int
           the total number of variables of the vector-valued function

        RETURNS
        ========
        jacobian: a numpy array with shape (functions_dim, vars_dim)
            the jabocian matrix of the vector-valued function

        NOTES
        =====
        PRE:
             - functions_array is a numpy array of only Ad_Var objects
             - the gradient vector of each function in functions_array must have dimensions equal vars_dim
               i.e. all functions in functions_array live in a space of equal dimensions.
        POST:
             - the values or the derivatives of the functions in functions_array are not changed
             - the result of get_jacobian is a numpy 2D array and not an Ad_Var object
             - raises a TypeError exception if any of the elements of the functions_array are not of type Ad_Var
             - raises a ValueError exception if the gradient of any of the elements of the functions_array have not length equal to vars_dim

        EXAMPLES
        =========
        >>> x = Ad_Var(1, np.array([1, 0]))
        >>> y = Ad_Var(2, np.array([0, 1]))
        >>> f = np.array([Ad_Var.cos(x) * (y + 2), 1 + x ** 2 / (x * y * 3), 3 * Ad_Var.log(x * 2) + Ad_Var.exp(x / y)])
        >>> Ad_Var.get_jacobian(f, 3, 2)
        array([[-3.36588394,  0.54030231],
               [ 0.16666667, -0.08333333],
               [ 3.82436064, -0.41218032]])
        """
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
        """
        Returns the values of for a vector-valued function evaluated at a point in higher dimensions.

        INPUTS
        =======
        functions_array: numpy array of Ad_Var objects
            a vector of functions which make up the vector-valued function together

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
        >>> x = Ad_Var(1, np.array([1, 0]))
        >>> y = Ad_Var(2, np.array([0, 1]))
        >>> f = np.array([Ad_Var.cos(x) * (y + 2), 1 + x ** 2 / (x * y * 3), 3 * Ad_Var.log(x * 2) + Ad_Var.exp(x / y)])
        >>> Ad_Var.get_values(f)
        array([2.16120922, 1.16666667, 3.72816281])
        """
        values = []
        for function in functions_array:
            if type(function).__name__ != 'Ad_Var':
                raise TypeError("The list of functions inputted is not a numpy array of Ad_Var objects.")
            values.append(function.get_val())
        return np.array(values)

    @staticmethod
    def grid_eval(func_string, vars_strings, Ad_Vars_list, grid):
        """
        Returns a dictionary with the function value and the corresponding derivative/gradient/jacobian
        evaluated at different points.
        INPUTS
        =======
        func_string: string
            if function is scalar: a string which is a function of already defined Ad_Var objects , e.g. "x + 2y"
            if function is vector-valued: a string which is a list of functions of Ad_Var objects, e.g. "[x+2y, x**y, Ad_Var.cos(x)]"
        vars_strings: list of strings
            a list of the string representations of each variable used in the func_string, used to create a custom scope using only the variables references in func_string
        Ad_Vars_list: list of Ad_Var objects
            a list of variables which are used in the function referenced in func_string
        grid: list of lists of points
            List of lists. The length of the outer list is equal to the the number of variables
            passed in the variables_list. For example, if the function is "x + 2*y*z", then the
            variables_list = [x, y, z]. If the grid = [[1,2], [3,4], [8]], then the user specifies
            that he wants to get the value and the gradient of the function at points (x,y,z) = (1,3,8),
            (1,4,8), (2,3,8), (2,4,8).
        RETURNS
        ========
        result_dict: dictionary
            Each key is a point on the grid passed and the value is a tuple with two elements. The first element of the tuple is
            the value of the function at this point, while the second element is the derivative (if function is scalar of one variable),
            gradient (if scalar function of multiple variables) or jacobian (if vector-valued function of multiple variables).
        NOTES
        =====
        PRE:
             - the string passed should be referencing already instantiated Ad_Var objects
             - the length of list with the string representations of variables should be equal to length of the list with the instantiated Ad_Var objects
             - Any elementary functions used in the string should be preceded with Ad_Var precisely, e.g. "Ad_Var.cos(x) + Ad_Var.exp(y)"
             - if the user wants to evaluate the jacobian on multiple points, the string should be a list of functions which
               reference already instantiated Ad_Var objects, e.g. "[x+2y, x**y, Ad_Var.cos(x)]"
             - The length of the variables_list should be equal to the length of the grid
        POST:
             - raises a ValueError if the func_string uses containts import statements
             - raises a ValueError if the length of list with the string representations of variables
               is not equal to length of the list with the instantiated Ad_Var objects
             - raises a ValueError if the length of the variables_list is not equal to the length of the grid. A list of values should
               be passed for each variable referenced in the function defined by the func_string.
        EXAMPLES
        =========
        >>> x = Ad_Var(1, np.array([1, 0]))
        >>> y = Ad_Var(2, np.array([0, 1]))
        >>> f_string = "[Ad_Var.cos(x) * (y + 2), 1 + x ** 2 / (x * y * 3), 3 * Ad_Var.log(x * 2) + Ad_Var.exp(x / y)]"
        >>> Ad_Var.grid_eval(f_string, ['x', 'y'], [x, y], [[1,2],[2,3]])
        {(1, 2): (array([2.16120922, 1.16666667, 3.72816281]), array([[-3.36588394,  0.54030231],
               [ 0.16666667, -0.08333333],
               [ 3.82436064, -0.41218032]])), (1, 3): (array([2.70151153, 1.11111111, 3.47505397]), array([[-4.20735492,  0.54030231],
               [ 0.11111111, -0.03703704],
               [ 3.46520414, -0.15506805]])), (2, 2): (array([-1.66458735,  1.33333333,  6.87716491]), array([[-3.63718971, -0.41614684],
               [ 0.16666667, -0.16666667],
               [ 2.85914091, -1.35914091]])), (2, 3): (array([-2.08073418,  1.22222222,  6.10661712]), array([[-4.54648713, -0.41614684],
               [ 0.11111111, -0.07407407],
               [ 2.14924468, -0.43282979]]))}
        >>> a = Ad_Var(1, 1)
        >>> f_string = "a**3"
        >>> Ad_Var.grid_eval(f_string, ['a'], [a], [[1,2,3]])
        {(1,): (1, 3), (2,): (8, 12), (3,): (27, 27)}
        """

        #avoid to evaluate anything that could be dangerous
        if "import" in func_string:
            raise ValueError("Function string can only be a sequence of operations on Ad_Var variables.")

        if len(vars_strings) != len(Ad_Vars_list):
            raise ValueError("Lengths of vars_strings and Ad_Vars_list should be equal.")

        #restrict the scope to variables used in the function
        scope = dict(zip(vars_strings, Ad_Vars_list))
        scope['Ad_Var'] = Ad_Var
        f = eval(func_string, scope)

        #make the list a numpy array
        if (func_string.startswith('[')) and (func_string.endswith(']')):
            f = np.array(f)

        if len(Ad_Vars_list) != len(grid):
            raise ValueError("Grid dimensions do not match with the number of variables used in function.")

        grid_points = list(product(*grid))
        result_dict = {}

        #if f is scalar function
        if type(f).__name__ == 'Ad_Var':
            for tuple in grid_points:
                for i, variable in enumerate(Ad_Vars_list):
                    variable.set_val(tuple[i])
                f = eval(func_string, scope)
                result_dict[tuple] = (f.get_val(), f.get_ders())

        #if f is a vector function
        else:
            for tuple in grid_points:
                for i, variable in enumerate(Ad_Vars_list):
                    variable.set_val(tuple[i])
                f = eval(func_string, scope)
                result_dict[tuple] = (Ad_Var.get_values(f), Ad_Var.get_jacobian(f, len(f), len(Ad_Vars_list)))

        return result_dict

class rAd_Var():
    def __init__(self, val):
        """
        Initialization of the rAd_Var class, with only the value of the variable passed in as an argument
        Parameters
        ==========
        self: rAd_Var
        val: variable value
        Examples
        =========
        >>> x = rAd_Var(3) #initializes an rAd_Var instance with value 3
        """

        # Set checks to ensure that only numeric values and arrays of single numeric values are accepted
        if type(val) is np.ndarray and len(val) == 1:
            self._val = val[0]

        elif isinstance(val, numbers.Number):
            self._val = val

        else:
            raise TypeError(f"{val} is invalid rAd_Var input!")

        self._ders = None
        self.parents = []
        self.children = []

        # Attribute used to track if a node has been visited in the backward pass
        # Set to True during tree traversal in get_ders(), then reset at end
        self.visited = False

    def __str__(self):
        """
        Returns Returns a string representing the value of `self._val` (Value) and the value of `self._ders` (Gradient)
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        Object representation for the rAd_Var instance with value and array of partial derivatives
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> print(x)
        Value = 3
        Partial Derivative(s) = [1]
        """
        print_stm = f'Value = {self._val}\nPartial Derivative(s) = {self.get_ders()}'
        return print_stm

    def __repr__(self):
        """
        Returns Returns a string representing the value of `self._val` (Value) and the value of `self._ders` (Gradient)
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        object representation for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> print(x)
        Value = 3
        Partial Derivative(s) = [1]
        """
        print_stm = f'Value = {self._val}\nPartial Derivative(s) = {self.get_ders()}'
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
        >>> x.get_gradient()
        4
        """
        self._ders = derivatives

    def get_ders(self):
        """
        Returns the partial derivative for the variables used in the rAd_Var instance passed
        by undertaking the backward pass on the node and its children
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        numpy array of partial derivative(s) for the rAd_Var instance at
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(1)
        >>> f = 2*x + 3*y
        >>> f.get_ders()
        array([2., 3.])
        """
        # Set the partial derivative of the final rAd_Var node as 1
        self.set_ders(1)

        input_var = [] # Create lists to track input variables
        seen_ids = [] # Create list of seen_ids to track ancestor nodes that have been visited
        gradient_matrix = np.array([]) # Instantiate empty gradient matrix to track gradients of input variables

        # Goes through the list of all ancestors of the final rAd_Var node
        for ancestor in self.get_ancestors():
            # Identify ancestor nodes as input variables if they do not have any parents and have not been included
            if not ancestor.parents and id(ancestor) not in seen_ids:
                input_var.append(ancestor)
                seen_ids.append(id(ancestor))

        # Obtain gradient for each input variable and append to gradient matrix
        for var in input_var:
            gradient_matrix = np.append(gradient_matrix, var.get_gradient())

        # Return partial derivative of final node if it does not have any additional input variables
        if input_var == []:
            return np.array([self._ders])

        return gradient_matrix

    def get_val(self):
        """
        Returns the value for the rAd_Var instance passed.
        Parameters
        ==========
        self: rAd_Var
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

    def _typecheck_other(self, other):
        if type(other) == Ad_Var:
            raise TypeError("rAd_Var object cannot be used with Ad_Var objects!")

    def __add__(self, other):
        """
        Returns the addition for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        addition for the rd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(1)
        >>> f = x + y
        >>> f.get_val()
        2
        >>> f.get_ders()
        array([1., 1.])
        """
        self._typecheck_other(other)
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
        """
        Returns the addition for one rAd_Var or numeric type instance and an rAd_Var instance passed.
        Parameters
        ==========
        self: rAd_Var/numeric type
        other: rAd_Var
        Returns
        =======
        addition for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(1)
        >>> f = x + y
        >>> f.get_val()
        2
        >>> f.get_ders()
        array([1., 1.])
        """
        self._typecheck_other(other)
        return self + other

    def __sub__(self, other):
        """
        Returns the subtraction for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        subtraction for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(1)
        >>> f = x - y
        >>> f.get_val()
        0
        >>> f.get_ders()
        array([ 1., -1.])
        """
        self._typecheck_other(other)
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
        """
        Returns the subtraction of one rAd_Var instance from another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        subtraction for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> f = 1 - x
        >>> f.get_val()
        0
        >>> f.get_ders()
        array([-1.])
        """
        self._typecheck_other(other)
        return - self + other

    def __mul__(self, other):
        """
        Returns the multiplication for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        rAd_Var instance that is the product of the two inputs
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(2)
        >>> f = x * y
        >>> f.get_val()
        2
        >>> f.get_ders()
        array([2., 1.])
        """
        self._typecheck_other(other)
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
        """
        Returns the multiplication for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        rAd_Var instance that is the product of the two inputs
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(2)
        >>> f = x * y
        >>> f.get_val()
        2
        >>> f.get_ders()
        array([2., 1.])
        """
        self._typecheck_other(other)
        return self * other

    def __truediv__(self, other):
        """
        Returns the division for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        division for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(2)
        >>> f = x / y
        >>> f.get_val()
        0.5
        >>> f.get_ders()
        array([ 0.5 , -0.25])
        """
        self._typecheck_other(other)
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
        """
        Returns the division for one rAd_Var instance and another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        division for the rAd_Var instance
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> f = 2 / x
        >>> f.get_val()
        2.0
        >>> f.get_ders()
        array([-2.])
        """
        self._typecheck_other(other)
        rad_object = rAd_Var(other / self._val)
        self.children.append((rad_object, - other / (self._val**2)))
        rad_object.parents = [self]
        return rad_object

    def __pow__(self, other):
        """
        Returns the power for one rAd_Var instance to another rAd_Var or numeric type instance passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        power for the rAd_Var instance, self ** other
        Examples
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(2)
        >>> f = x ** y
        >>> f.get_val()
        1
        >>> f.get_ders()
        array([2., 0.])
        """
        self._typecheck_other(other)
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
        """
        Returns the power for one rAd_Var instance or numeric type instance to another rAd_Var passed.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var/numeric type
        Returns
        =======
        power for the rAd_Var instance, self ** other
        Examples
        =========
        >>> x = 2
        >>> y = rAd_Var(2)
        >>> f = x ** y
        >>> f.get_val()
        4
        >>> f.get_ders()
        array([2.77258872])
        """
        self._typecheck_other(other)
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
        self: rAd_Var
        other: rAd_Var
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
        self._typecheck_other(other)
        if self._val == other._val and self.get_gradient() == other.get_gradient():
            return True
        else:
            return False

    def __ne__(self, other):
        """
        Returns whether the two rAd_Var instances passed do not have the same values and derivatives.
        Parameters
        ==========
        self: rAd_Var
        other: rAd_Var
        Returns
        =======
        inequality check for two rAd_Var instances
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
        self._typecheck_other(other)
        return not self == other

    def __neg__(self):
        """
        Returns the negation for the rAd_Var instance passed.
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        negation for the rAd_Var instance
        Examples
        =========
        >>> x = - rAd_Var(3)
        >>> x
        Value = -3
        Partial Derivative(s) = [-1.]
        """
        rad_object = rAd_Var(self._val * -1)
        self.children.append((rad_object, -1))
        rad_object.parents = [self]
        return rad_object

    def sqrt(self):
        """
        Returns the square root of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the square root of the object passed
        Examples
        =========
        >>> x = rAd_Var(16)
        >>> f = rAd_Var.sqrt(x)
        >>> f.get_val()
        4.0
        >>> f.get_ders()
        array([0.125])
        """
        rad_object = rAd_Var(self._val ** 0.5)
        self.children.append((rad_object, 0.5 * self._val ** (-0.5)))
        rad_object.parents = [self]
        return rad_object

    def exp(self):
        """
        Returns the exponential of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the exponential of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(1)
        >>> x2 = rAd_Var(2)
        >>> f = rAd_Var.exp(x1 + x2)
        >>> f.get_val()
        20.085536923187668
        >>> f.get_ders()
        array([20.08553692, 20.08553692])
        """
        rad_object = rAd_Var(np.exp(self._val))
        self.children.append((rad_object, np.exp(self._val)))
        rad_object.parents = [self]
        return rad_object

    def log(self, logbase=np.e):
        """
        Returns the logarithm of the rAd_Var instance passed with the base specified by the user
        Parameters
        ==========
        self: rAd_Var
        logbase: float, optional, default value is np.e
                 the base of the logarithm
        Returns
        =======
        rAd_Var object which is the logarithm with base logbase of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(2)
        >>> x2 = rAd_Var(3)
        >>> f = rAd_Var.log(x1 + 2*x2)
        >>> f.get_val()
        2.0794415416798357
        >>> f.get_ders()
        array([0.125, 0.25 ])
        """
        rad_object = rAd_Var(np.log(self._val))
        self.children.append((rad_object, 1/(self._val*np.log(logbase))))
        rad_object.parents = [self]
        return rad_object

    def logistic(self):
        """
        Returns the logistic function evaluated at value of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which has value equal to sigmoid of the value of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(0)
        >>> f = rAd_Var.logistic(x1)
        >>> f.get_val()
        0.5
        >>> f.get_ders()
        array([0.25])
        """
        rad_object = rAd_Var(1/(1+np.exp(-self._val)))
        self.children.append((rad_object, (np.exp(-self._val)/((np.exp(-self._val)+1)**2))))
        rad_object.parents = [self]
        return rad_object


    def sin(self):
        """
        Returns the sine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the sine of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(np.pi/4)
        >>> f = rAd_Var.sin(x1)
        >>> f.get_val()
        0.7071067811865476
        >>> f.get_ders()
        array([0.70710678])
        """
        rad_object = rAd_Var(np.sin(self._val))
        self.children.append((rad_object, (np.cos(self._val))))
        rad_object.parents = [self]
        return rad_object

    def cos(self):
        """
        Returns the cosine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the cosine of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(np.pi/4)
        >>> f = rAd_Var.cos(x1)
        >>> f.get_val()
        0.7071067811865476
        >>> f.get_ders()
        array([-0.70710678])
        """
        rad_object = rAd_Var(np.cos(self._val))
        self.children.append((rad_object, -(np.sin(self._val))))
        rad_object.parents = [self]
        return rad_object

    def tan(self):
        """
        Returns the tangent of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the tangent of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(0)
        >>> f = rAd_Var.tan(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        array([1.])
        """
        rad_object = rAd_Var(np.tan(self._val))
        self.children.append((rad_object, 1/(np.cos(self._val))**2))
        rad_object.parents = [self]
        return rad_object

    def arcsin(self):
        """
        Returns the arcsin of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the arcsin of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(0)
        >>> f = rAd_Var.arcsin(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        array([1.])
        """
        if -1 <= self._val <= 1:
            rad_object = rAd_Var(np.arcsin(self._val))
            self.children.append((rad_object, 1/np.sqrt(1 - (self._val ** 2))))
            rad_object.parents = [self]
            return rad_object
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def arccos(self):
        """
        Returns the arccos of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the arccos of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(0)
        >>> f = rAd_Var.arccos(x1)
        >>> f.get_val()
        1.5707963267948966
        >>> f.get_ders()
        array([-1.])
        """
        if -1 <= self._val <= 1:
            rad_object = rAd_Var(np.arccos(self._val))
            self.children.append((rad_object, -1/np.sqrt(1 - (self._val ** 2))))
            rad_object.parents = [self]
            return rad_object
        else:
            raise ValueError('The domain of the inverse trig function should be [-1,1]')

    def arctan(self):
        """
        Returns the arctan of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the arctan of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(0)
        >>> f = rAd_Var.arctan(x1)
        >>> f.get_val()
        0.0
        >>> f.get_ders()
        array([1.])
        """
        rad_object = rAd_Var(np.arctan(self._val))
        self.children.append((rad_object, 1/(1 + self._val ** 2)))
        rad_object.parents = [self]
        return rad_object

    def sinh(self):
        """
        Returns the hyperbolic sine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the hyperbolic sine of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(np.pi/4)
        >>> f = rAd_Var.sinh(x1)
        >>> f.get_val()
        0.8686709614860095
        >>> f.get_ders()
        array([1.32460909])
        """
        rad_object = rAd_Var(np.sinh(self._val))
        self.children.append((rad_object, np.cosh(self._val)))
        rad_object.parents = [self]
        return rad_object

    def cosh(self):
        """
        Returns the hyperbolic cosine of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the hyperbolic cosine of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(np.pi/4)
        >>> f = rAd_Var.cosh(x1)
        >>> f.get_val()
        1.324609089252006
        >>> f.get_ders()
        array([0.86867096])
        """
        rad_object = rAd_Var(np.cosh(self._val))
        self.children.append((rad_object, np.sinh(self._val)))
        rad_object.parents = [self]
        return rad_object

    def tanh(self):
        """
        Returns the hyperbolic tangent of the Ad_Var instance passed.
        Parameters
        ==========
        self: Ad_Var
        Returns
        =======
        Ad_Var object which is the hyperbolic tangent of the object passed
        Examples
        =========
        >>> x1 = rAd_Var(np.pi/4)
        >>> f = rAd_Var.tanh(x1)
        >>> f.get_val()
        0.6557942026326724
        >>> f.get_ders()
        array([0.56993396])
        """
        rad_object = rAd_Var(np.tanh(self._val))
        self.children.append((rad_object, (1 - np.tanh(self._val)**2)))
        rad_object.parents = [self]
        return rad_object

    @staticmethod
    def get_jacobian(functions_array, var_list, var_values):
        """
        Returns the jacobian matrix for a vector of functions, with given values for variables in the function
        INPUTS
        =======
        functions_array: numpy array of Python function
            a vector of functions passed into the method
        var_list: numpy array of strings
            variable names for variables in the functions array
        var_values: numpy array of numeric values
           values for variables in the functions array
        RETURNS
        ========
        jacobian: a numpy array with shape (len(functions_array), len(var_values)), the jacobian matrix of the vector-valued function
        NOTES
        =====
        PRE:
             - functions_array is a numpy array of Python functions
             - the gradient vector of each function in functions_array must have dimensions equal to vars_dim
               i.e. all functions in functions_array live in a space of equal dimensions.
        POST:
             - the result of get_jacobian is a numpy 2D array and not an rAd_Var object
             - raises a ValueError exception if the number of arguments required for a function is greater than the number of input variables
        EXAMPLES
        =========
        >>> def f1(x, y):
        ...  return x * y
        >>> def f2(x,y):
        ...  return x + y
        >>> def f3(x, y):
        ...  return rAd_Var.log(x ** y)
        >>> rAd_Var.get_jacobian([f1, f2, f3], ["x", "y"], [1, 2])
        array([[2., 1.],
               [1., 1.],
               [2., 0.]])
        """

        #input is a numpy array of rAd_Var function
        functions_dim = len(functions_array)
        vars_dim = len(var_values)

        jacobian = np.zeros((functions_dim, vars_dim))
        list_partial_ders = []

        # Raise error if the number of input variables does not match the value numbers
        if len(var_list) != len(var_values):
            raise ValueError(f"Number of input variables does not match the number of input values.")


        # Create dictionary of variables to their input values
        variable_value_dict = {}
        for var, value in zip(var_list, var_values):
            variable_value_dict[var] = value

        # For the list of functions, create rAd_Var instances for variables used in the function
        for i, function in enumerate(functions_array):
            func_variable = {}
            func_variable_list = list(function.__code__.co_varnames)

            for var in func_variable_list:
                if var not in variable_value_dict:
                    raise ValueError("The variable required as input for your function is not defined in the constructor.")
                func_variable[var] = rAd_Var(variable_value_dict[var])

            partial_der = function(**func_variable).get_ders()

            dict_partial_der = {}
            for variable, der in zip(func_variable_list, partial_der):
                dict_partial_der[variable] = der

            list_partial_ders.append(dict_partial_der)

        #Get a full list of all variables from the dictionary
        #Map the variable names to column number in the Jacobian
        col_dict = {}
        for index, var in enumerate(var_list):
            col_dict[index] = var

        #For each row in the jacobian matrix, assign values based on variable names; if it does not exist, assign 0
        for i in range(jacobian.shape[0]):
            partial_der = list_partial_ders[i]

            for j in range(jacobian.shape[1]):
                var_name = col_dict[j]
                jacobian[i][j] = 0 if var_name not in partial_der else partial_der[var_name]

        return jacobian

    @staticmethod
    def get_values(functions_array, var_list, var_values):
        """
        Returns the values of for a vector-valued function evaluated at a point in higher dimensions.
        INPUTS
        =======
        functions_array: numpy array of Python function
            a vector of functions passed into the method
        var_list: numpy array of strings
            variable names for variables in the functions array
        var_values: numpy array of numeric values
           values for variables in the functions array
        RETURNS
        ========
        values: a numpy array with shape (functions_dim,)
            the vector with the values of the vector-valued function evaluated at a point
        NOTES
        =====
        PRE:
             - functions_array is a numpy array of only rAd_Var objects
        POST:
             - raises a TypeError exception if any of the elements of the functions_array are not of type rAd_Var
        EXAMPLES
        =========
        >>> def f1(x, y):
        ...  return x * y
        >>> def f2(x,y):
        ...  return x + y
        >>> def f3(x, y):
        ...  return rAd_Var.log(x ** y)
        >>> rAd_Var.get_values([f1, f2, f3], ["x", "y"], [1, 2])
        array([2., 3., 0.])
        """
        values = []

        # Raise error if the number of input variables does not match the value numbers
        if len(var_list) != len(var_values):
            raise ValueError(f"Number of input variables does not match the number of input values.")

        # Create dictionary of variables to their input values
        variable_value_dict = {}
        for var, value in zip(var_list, var_values):
            variable_value_dict[var] = value

        # For the list of functions, create rAd_Var instances for variables used in the function
        for i, function in enumerate(functions_array):
            func_variable = {}
            func_variable_list = list(function.__code__.co_varnames)

            for var in func_variable_list:
                if var not in variable_value_dict:
                    raise ValueError("The variable required for your function is not defined in the constructor.")
                func_variable[var] = rAd_Var(variable_value_dict[var])

            values.append(function(**func_variable).get_val())

        return np.array(values)

    def get_ancestors(self):
        """
        Helper function to return a list of ancestors for an rAd_Var object, used in the backward pass
        INPUTS
        =======
        self: rAd_Var
        RETURNS
        ========
        ancestorlist: a list of all ancestor (parents, parents of parents etc.) nodes of the final rAd_Var object
        EXAMPLES
        =========
        >>> x = rAd_Var(1)
        >>> y = rAd_Var(3)
        >>> f = x + y
        >>> f.get_ancestors()
        [Value = 1
        Partial Derivative(s) = [1], Value = 3
        Partial Derivative(s) = [1]]
        """
        ancestorlist = []
        if self.parents:
            for parent in self.parents:
                if not parent.visited:
                    ancestorlist.append(parent)
                    ancestorlist += parent.get_ancestors()
                    parent.visited = True

        # Reset all nodes in tree as being unseen for future traversals
        for ancestor in ancestorlist:
            ancestor.visited = False

        return ancestorlist

    def get_gradient(self):
        """
        Helper method to return the gradient in the backward pass of the reverse mode, used in the Jacobian matrix
        Parameters
        ==========
        self: rAd_Var
        Returns
        =======
        Gradient for the rAd_Var instance obtained through the backward pass
        Examples
        =========
        >>> x = rAd_Var(3)
        >>> x.get_gradient()
        0
        """
        if self._ders is None:
            new_deriv = sum(weight * var.get_gradient() for var, weight in self.children)
            self.set_ders(new_deriv)
        return self._ders