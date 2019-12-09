# CS207 Final Project
Group 11 - The Differentiators

Michael Scott, Dimitris Vamvourellis, Yiwen Wang, Royce Yap

[![Build Status](https://travis-ci.org/the-differentiators/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/the-differentiators/cs207-FinalProject)

[![Coverage Status](https://codecov.io/gh/the-differentiators/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/the-differentiators/cs207-FinalProject)

### How to install `ADKit`

ADKit can be installed through the Python Package Index using the following command in the command terminal:

    pip install ADKit

Alternatively, the user may install ADKit by cloning the github repository (https://github.com/the-differentiators/cs207-FinalProject.git) or downloading as a zipped archive.

ADKit has only `numpy` (v. 1.14.3 or higher) as a pre-installation requirement. If `numpy` is not installed, this can be installed using the `requirements.txt` file included in the repository, after it is downloaded, using the following code:

    pip install -r requirements.txt

### How to use `ADKit` (Forward Mode)

The following steps walk the user through a basic demo of how to import and use the `ADKit` package for a scalar multivariate function:

#### Importing `ADKit.AutoDiff` and requirements
The following code imports the forward mode variable class from ADKit.  

    from ADKit.AutoDiff import Ad_Var

#### Using `ADKit` to compute derivative of a vector-valued multivariate function (forward mode)

The user can also use `ADKit` to calculate the value and the jacobian matrix of a vector-valued function. Again the variables must be instantiated in the same way as discussed above. Then, a vector-valued function can be defined as a numpy array of functions composed of instantiated `Ad_Var` variables. 

An example is shown below for the vector valued function `f = [sin^2(2x) + z^y][e^x + z]` for `x = 1, y = 2, z = 3`: 

    x = Ad_Var(1, np.array([1, 0, 0]))
    y = Ad_Var(2, np.array([0, 1, 0]))
    z = Ad_Var(3, np.array([0, 0, 1]))

    f = np.array([(Ad_Var.sin(2*x))**2 + z**y, Ad_Var.exp(x) + z])

Then, the user can call `get_jacobian` to get the jacobian matrix of `f` evaluated at `x = 1, y = 2, z = 3`. The first argument of this method is the vector-valued function `f` defined as a numpy array. The second argument is the dimension of the vector of the functions (in this example the vector-valued function has 2 dimensions). The third argument is the number of variables composing the vector-valued function (in this example vector-valued function is composed of 3 variables, `x,y` and `z`).

    Ad_Var.get_jacobian(f, 2, 3)

Also, the user can call `get_values` by passing `f`, to calculate the value of the vector-valued function for the given values of the variables.

    Ad_Var.get_values(f)

For more information on the forward mode and the extension (Reverse Mode), please refer to the Documentation file in the `docs` folder.
