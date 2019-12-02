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

    def __eq__(self, other):
        """
        Returns whether the two the Ad_Var instances passed have the same values and derivatives.
        
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
        return self._val == other._val and self._ders == other._ders

    def __ne__(self, other):
        """
        Returns whether the two the Ad_Var instances passed do not have the same values and derivatives.
        
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
        return self._val != other._val or self._ders != other._ders

    def __repr__(self):
        """
        Returns the object representation for the Ad_Var instance passed.
        
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
        2
        >>> f.get_ders()
        1
        """
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
        # 3/Ad_Var(4)
        return Ad_Var(other / self._val, - self._ders*other / (self._val) ** 2)
        '''except AttributeError:
            return Ad_Var(other / self._val , - other / (self._ders)**2)'''

    def __pow__(self, other):
        """
        Returns the power for one Ad_Var instance to the another Ad_Var or numeric type instance passed.
        
        Parameters
        ==========
        self: Ad_Var
        other: Ad_Var/numeric type
        
        Returns
        =======
        power for the Ad_Var instance
        
        Examples
        =========
        >>> x = Ad_Var(1,2)
        >>> y = Ad_Var(2,1)
        >>> f = x ** y
        >>> f.get_val()
        1
        >>> f.get_ders()
        4
        """
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
        20.085536923187668
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
        [0.125, 0.25]
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
        0.7071067811865475
        >>> f.get_ders()
        0.7071067811865475
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
        0.7071067811865475
        >>> f.get_ders()
        -0.7071067811865475
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
        return Ad_Var(np.arcsin(self._val), self._ders / np.sqrt(1 - (self._val ** 2))) 

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
        return Ad_Var(np.arccos(self._val), -self._ders / np.sqrt(1 - (self._val ** 2))) 

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
        0
        >>> f.get_ders()
        1.0
        """
        return Ad_Var(np.arctan(self._val), self._ders / (1 + self._val ** 2))

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
        [ 0.10499359 -0.20998717]
        """
        def sigmoid(x):
            return 1.0/(1 + np.exp(-x))

        def sigmoid_derivative(x):
            der = (1 - sigmoid(x)) * sigmoid(x)
            return der

        return Ad_Var(sigmoid(self._val), sigmoid_derivative(self._val) * self._ders)

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
        1.3246090892520057
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
        1.3246090892520057
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
        0.24541076067097178
        """
        return Ad_Var(np.tanh(self._val), self._ders*(1 - np.sinh(self._val)**2))



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
        [[-3.36588394  0.54030231]
        [ 0.16666667 -0.08333333]
        [ 3.82436064 -0.41218032]]
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
        [2.16120922 1.16666667 3.72816281]
        """
        values = []
        for function in functions_array:
            if type(function).__name__ != 'Ad_Var':
                raise TypeError("The list of functions inputted is not a numpy array of Ad_Var objects.")
            values.append(function.get_val())
        return np.array(values)

    @staticmethod
    def grid_eval(func_string, variables_list, grid):
        f = eval(func_string)

        if len(variables_list) != len(grid):
            raise ValueError("Grid dimensions do not match with the number of variables used in function.")

        grid_points = list(product(*grid))
        result_dict = {}

        #if f is scalar function
        if type(f).__name__ == 'Ad_Var':
            for tuple in grid_points:
                for i, variable in enumerate(variables_list):
                    variable.set_val(tuple[i])
                f = eval(func_string)
                result_dict[tuple] = (f.get_val(), f.get_ders())

        #if f is a vector function
        else:
            for tuple in grid_points:
                for i, variable in enumerate(variables_list):
                    variable.set_val(tuple[i])
                f = eval(func_string)
                result_dict[tuple] = (Ad_Var.get_values(f), Ad_Var.get_jacobian(f, len(f), len(variables_list)))

        return result_dict








if __name__=='__main__':
    x = Ad_Var(1, np.array([1, 0]))
    y = Ad_Var(2, np.array([0, 1]))
    f_string = "np.array([Ad_Var.cos(x) * (y + 2), 1 + x ** 2 / (x * y * 3), 3 * Ad_Var.log(x * 2) + Ad_Var.exp(x / y)])"
    dict = Ad_Var.grid_eval(f_string, [x, y], [[1,2],[2,3]])
    print(dict)
    a = Ad_Var(1, 1)
    f_string = "2*a"
    dict1 = Ad_Var.grid_eval(f_string, [a], [[1,2,3]])
    print(dict1)
    f = 2*a