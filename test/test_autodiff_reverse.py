#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../')

from src.AutoDiff import rAd_Var, Ad_Var

def test_exp():
    # Scalar
    a = rAd_Var(2)
    f = a.exp()
    ders = f.get_ders()
    assert(f.get_val() == np.exp(2))
    assert(ders == [np.exp(2)])
    # Gradient
    x = rAd_Var(3)
    y = rAd_Var(-4)
    x1 = rAd_Var.exp(x + y)
    ders = x1.get_ders()
    assert(x1.get_val() == np.exp(-1))
    assert(ders == [np.exp(-1), np.exp(-1)]).all()

def test_add():
    # Scalar
    a, b = rAd_Var(99), rAd_Var(99)
    f = a + 1
    g = 1 + b
    ders_f = f.get_ders()
    ders_g = g.get_ders()
    assert(ders_f == ders_g and f.get_val() == g.get_val())
    assert(f.get_val() == 100)
    assert(ders_f == [1])
    # Gradient
    x = rAd_Var(4)
    y = rAd_Var(8)
    z = rAd_Var(-2)
    x1 = x + y + z
    ders = x1.get_ders()
    assert(x1.get_val() == 10)
    assert(ders == [1, 1, 1]).all()

def test_sub():
    # Scalar
    a = rAd_Var(101)
    b = 1
    f = a - b
    ders = f.get_ders()
    assert(f.get_val() == 100)
    assert(ders == [1])
    # Rsub
    c = rAd_Var(50)
    d = 100
    g = d - c
    ders = g.get_ders()
    assert(g.get_val() == 50)
    assert(ders == -1)
    # Gradient
    x = rAd_Var(500)
    y = rAd_Var(100)
    z = rAd_Var(-100)
    x1 = x - y - z
    ders = x1.get_ders()
    assert(x1.get_val() == 500)
    assert(ders == [1, -1, -1]).all()

def test_mul():
    # Scalar
    a = rAd_Var(12)
    b = 3
    f = a * b
    c = rAd_Var(12)
    g = b * c
    ders_f = f.get_ders()
    ders_g = g.get_ders()
    assert(f.get_val() == g.get_val() == 36)
    assert(ders_f == ders_g) 
    assert(ders_f == [3]).all()
    # Gradient
    x = rAd_Var(3)
    y = rAd_Var(-2)
    z = rAd_Var(3)
    x1 = x * y * z
    ders = x1.get_ders()
    assert(x1.get_val() == -18)
    assert(ders == [-6, 9, -6]).all()

def test_div():
    # Scalar
    a = rAd_Var(20)
    b = 2
    f = a / b
    c = rAd_Var(20)
    g = b / c
    ders_f = f.get_ders()
    ders_g = g.get_ders()
    assert(f.get_val() == 10)
    assert(g.get_val() == 0.1)
    assert(ders_f == 1/2)
    assert(ders_g == -1/200)
    # Gradient
    x = rAd_Var(10)
    y = rAd_Var(5)
    x1 = x / y
    ders = x1.get_ders()
    assert(x1.get_val() == 2)
    assert(ders == [1/5, -2/5]).all()

def test_log():
    # Scalar
    a = rAd_Var(5)
    b = a.log()
    ders = b.get_ders()
    assert(b.get_val() == np.log(5))
    assert(ders == .2)
    c = rAd_Var(np.e)
    d = c.log()
    ders = d.get_ders()
    assert(d.get_val() == 1)
    assert(ders == 1/np.e)

def test_pow():
    # Scalar
    a = rAd_Var(2)
    b = 3
    f = a ** b
    ders = f.get_ders()
    assert(f.get_val() == 8)
    assert(ders == [12])
    # Type test.
    try:
        _ = a ** "NaN"
    except:
        print("test_pow type checking successful!")
    # Rpow.
    c = rAd_Var(4)
    d = 3
    g = d ** c
    ders_g = g.get_ders()
    assert(g.get_val() == 81)
    assert(ders_g == [81*np.log(3)])
    # Rpow bad input test.
    try:
        _ = 'NaN' ** c
    except:
        print("Rpow non-numeric base test passed!")
    # Gradient
    x = rAd_Var(2)
    y = rAd_Var(2)
    z = rAd_Var(2)
    x1 = x ** y ** z
    ders = x1.get_ders()
    assert(x1.get_val() == 16)
    assert(ders == [32, 64*np.log(2), 64*(np.log(2) ** 2)]).all()

def test_eq():
    a = rAd_Var(1)
    b = rAd_Var(1)
    c = rAd_Var(2)
    assert(a == b and a != c)

def test_neg():
    a = rAd_Var(5)
    f = -a
    ders = f.get_ders()
    assert(f.get_val() == -5)
    assert(ders == -1)

def test_sqrt():
    a = rAd_Var(16)
    f = a.sqrt()
    ders = f.get_ders()
    assert(f.get_val() == 4)
    assert(ders == 1/8)

def test_trig():
    # Scalar
    b = rAd_Var.sin(rAd_Var(np.pi/4))
    c = rAd_Var.cos(rAd_Var(np.pi/4))
    d = rAd_Var(np.pi)
    e = rAd_Var.tan(d)
    ders_b = b.get_ders()
    ders_c = c.get_ders()
    ders_e = e.get_ders()
    assert(b.get_val() == np.sin(np.pi/4))
    np.testing.assert_almost_equal(ders_b, np.cos(np.pi/4))
    assert(c.get_val() == np.cos(np.pi/4))
    np.testing.assert_almost_equal(ders_c, -np.sin(np.pi/4))
    assert(e.get_val() == np.tan(np.pi))
    np.testing.assert_almost_equal(ders_e, 1/np.cos(np.pi)**2)
    a1 = rAd_Var(np.pi)
    a2 = a1.sin()
    a2.get_ders()
    # Gradient
    x = rAd_Var(np.pi/4).sin()
    y = rAd_Var(np.pi/4).cos()
    z = rAd_Var(np.pi/4).tan()
    x1 = x + y + z
    ders = x1.get_ders()
    assert(x1.get_val() == 2**0.5 + 1)
    np.testing.assert_array_almost_equal(ders, [np.cos(np.pi/4), -np.sin(np.pi/4), 1/np.cos(np.pi/4)**2])


def test_inverse_trig():
    # Improper input test
    try:
        _ = rAd_Var(-2).arcsin()
    except ValueError:
        print("Arcsin domain test passed!")
    try:
        _ = rAd_Var(-2).arccos()
    except ValueError:
        print("Arcos domain test passed!")
    # Scalar
    a1, a2, a3 = rAd_Var(0.1), rAd_Var(0.1), rAd_Var(0.1)
    b = rAd_Var.arcsin(a1)
    c = rAd_Var.arccos(a2)
    d = rAd_Var.arctan(a3)
    ders_b = b.get_ders()
    ders_c = c.get_ders()
    ders_d = d.get_ders()
    assert b.get_val() == np.arcsin(0.1)
    assert ders_b == 1/np.sqrt(1-0.1**2)
    assert c.get_val() == np.arccos(0.1)
    assert ders_c == -1/np.sqrt(1-0.1**2)
    assert d.get_val() == np.arctan(0.1)
    assert ders_d == 1/(1+0.1**2)
    # Gradient
    x1 = rAd_Var(0.1)
    x2 = rAd_Var(0.2)
    x3 = rAd_Var(0.3)
    f = rAd_Var.arcsin(x1) + rAd_Var.arccos(x2) + rAd_Var.arctan(x3)
    ders = f.get_ders()
    assert f.get_val() == 1.7610626216439926
    assert (ders == [1.005037815259212, -1.0206207261596576, 0.9174311926605504]).all()

def test_logistic():
    def sigmoid(x):
        return 1.0/(1 + np.exp(-x))

    def sigmoid_derivative(x):
        der = (1 - sigmoid(x)) * sigmoid(x)
        return der

    a = rAd_Var(1)
    b = rAd_Var.logistic(a)
    ders = b.get_ders()
    assert b.get_val() == sigmoid(1)
    np.testing.assert_almost_equal(ders, sigmoid_derivative(1))
    x1 = rAd_Var(0.1)
    x2 = rAd_Var(0.2)
    f = rAd_Var.logistic(x1) - rAd_Var.logistic(x2)
    ders = f.get_ders()
    assert f.get_val() == sigmoid(0.1) - sigmoid(0.2)
    np.testing.assert_array_almost_equal(ders, [sigmoid_derivative(0.1), -sigmoid_derivative(0.2)])

def test_hyperbolic():
    # Scalar
    a1, a2, a3 = rAd_Var(0.1), rAd_Var(0.1), rAd_Var(0.1)
    b = rAd_Var.sinh(a1)
    c = rAd_Var.cosh(a2)
    d = rAd_Var.tanh(a3)
    ders_b = b.get_ders()
    ders_c = c.get_ders()
    ders_d = d.get_ders()
    assert b.get_val() == np.sinh(0.1)
    assert ders_b == np.cosh(0.1)
    assert c.get_val() == np.cosh(0.1)
    assert ders_c == np.sinh(0.1)
    assert d.get_val() == np.tanh(0.1)
    assert ders_d == (1 - np.tanh(0.1)**2)
    # Gradient
    x1 = rAd_Var(0.1)
    x2 = rAd_Var(0.2)
    x3 = rAd_Var(0.3)
    f = rAd_Var.sinh(x1) + rAd_Var.cosh(x2) - rAd_Var.tanh(x3)
    ders = f.get_ders()
    assert f.get_val() == np.sinh(0.1) + np.cosh(0.2) - np.tanh(0.3)
    np.testing.assert_array_almost_equal(ders, [np.cosh(0.1), np.sinh(0.2), -1+np.tanh(0.3)**2])

def test_multi_parent():
    x = rAd_Var(1)
    y = rAd_Var(2)
    x1 = x * y
    x2 = x1.exp()
    x3 = x1 + x2
    ders = x3.get_ders()
    np.testing.assert_almost_equal(x3.get_val(), 2 + (np.e ** 2))
    np.testing.assert_array_almost_equal(ders, [2+(2 * np.e ** 2), 1 + (np.e ** 2)])

def test_input():
    try:
        print(rAd_Var('NaN'))
    except TypeError:
        print("Input test 1 passed.")
    try:
        print(rAd_Var(None))
    except TypeError:
        print("Input test 2 passed.")
    try:
        _ = rAd_Var(5) + Ad_Var(5)
    except TypeError:
        print("rAd_Var and Ad_Var incompatibility check passed.")
    a = rAd_Var(np.array([42]))
    a.set_val(5)
    assert(a.get_val() == 5)
    print("Testing for __str__ method:\n", a)
    repr(a)

def test_jacobian():
    def f1(x, y):
        return rAd_Var.cos(x) * (y + 2)
    def f2(x, y):
        return 1 + x ** 2 / (x * y * 3)
    def f3(x, y):
        return 3 * rAd_Var.log(x * 2) + rAd_Var.exp(x / y)
    np.testing.assert_array_almost_equal(rAd_Var.get_jacobian([f1, f2, f3], ["x", "y"], [1, 2]), [[-3.36588394, 0.54030231],[0.16666667, -0.08333333],[3.82436064, -0.41218032]])

def test_jacobian2():
    def f1(x, y, z, a):
        return y + 2*x + z + a
    def f2(x, z):
        return 3 * x + z
    def f3(x, y):
        return rAd_Var.log(x ** y)

    np.testing.assert_array_almost_equal(rAd_Var.get_jacobian([f1, f2, f3], ["x", "y", "z", "a"], [1, 2, 3, 4]),[[1 , 2, 1, 1], [3, 0 , 1, 0],[2, 0 , 0 ,0 ]])

def test_jacobian_input():
    def f1(x, y, z, a):
        return y + 2*x + z + a
    def f2(x, z):
        return 3 * x + z
    def f3(x, y):
        return rAd_Var.log(x ** y)

    try:
        rAd_Var.get_jacobian([f1, f2, f3], ["x", "y", "z", "a"], [1, 2, 3])
    except:
        print ("test_jacobian_input: Caught input error where variables > values")

    try:
        rAd_Var.get_jacobian([f1, f2, f3], ["x"], [1])
    except:
        print ("test_jacobian_input: Caught input error where variables required for function are not defined")

def test_get_val():
    def f1(x, y):
        return rAd_Var.cos(x) * (y + 2)
    def f2(x, z):
        return 1 + x ** 2 / (x * z * 3)
    def f3(x, y):
        return 3 * rAd_Var.log(x * 2) + rAd_Var.exp(x / y)
    np.testing.assert_array_almost_equal(rAd_Var.get_values([f1, f2, f3], ["x", "z", "y"], [1, 3, 2]), np.array([2.161209, 1.111111, 3.728163]))

def test_get_val_input():
    def f1(x, y, z, a):
        return y + 2*x + z + a
    def f2(x, z):
        return 3 * x + z
    def f3(x, y):
        return rAd_Var.log(x ** y)

    try:
        rAd_Var.get_values([f1, f2, f3], ["x", "y", "z", "a"], [1, 2, 3])
    except:
        print ("test_get_val_input: Caught input error where variables > values")

    try:
        rAd_Var.get_values([f1, f2, f3], ["x"], [1])
    except:
        print ("test_get_val_input: Caught input error where variables required for function are not defined")

test_exp()
test_add()
test_sub()
test_mul()
test_div()
test_pow()
test_eq()
test_neg()
test_sqrt()
test_trig()
test_inverse_trig()
test_input()
test_log()
test_logistic()
test_hyperbolic()
test_jacobian()
test_jacobian2()
test_jacobian_input()
test_get_val()
test_get_val_input()
test_multi_parent()
print("All tests passed!")