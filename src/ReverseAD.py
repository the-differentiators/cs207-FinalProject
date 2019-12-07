import numpy as np
import numbers as numbers

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
            raise TypeError()

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
             - functions_array is a numpy array of only rAd_Var objects
             - the gradient vector of each function in functions_array must have dimensions equal to vars_dim
               i.e. all functions in functions_array live in a space of equal dimensions.
        POST:
             - the values or the derivatives of the functions in functions_array are not changed
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
        >>> rAd_Var.get_jacobian([f1, f2, f3], [1, 2])
        array([[2., 1.],
               [1., 1.],
               [2., 0.]])
        """

        #input is a numpy array of rAd_Var function
        functions_dim = len(functions_array)
        vars_dim = len(var_values)

        jacobian = np.zeros((functions_dim, vars_dim))
        for i, function in enumerate(functions_array):
            variables = []
            for value in var_values:
                variables.append(rAd_Var(value))
            if len(function.__code__.co_varnames) > len(variables):
                raise ValueError(f"Number of arguments required for function is greater than the number of input variables ({vars_dim}).")
            jacobian[i] = function(*variables).get_ders()

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

if __name__=='__main__':
    import doctest
    doctest.testmod(verbose = False)
    print("Passed all doctests!")