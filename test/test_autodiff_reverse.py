#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
sys.path.append('../')

from src.ReverseAD import rAd_Var

def test_exp():
    # Scalar
    a = rAd_Var(2)
    f = a.exp()
    ders = f.runreverse()
    assert(f.get_val() == np.exp(2))
    assert(ders == [np.exp(2)])
    # Gradient
    x = rAd_Var(3)
    y = rAd_Var(-4)
    x1 = rAd_Var.exp(x + y)
    ders = x1.runreverse()
    assert(x1.get_val() == np.exp(-1))
    assert(ders == [np.exp(-1), np.exp(-1)]).all()

def test_add():
    # Scalar
    a = rAd_Var(99)
    f = a + 1
    g = 1 + a
    ders_f = f.runreverse()
    ders_g = g.runreverse()
    assert(ders_f == ders_g and f.get_val() == g.get_val())
    assert(f.get_val() == 100)
    assert(ders_f == [1])
    # Gradient
    x = rAd_Var(4)
    y = rAd_Var(8)
    z = rAd_Var(-2)
    x1 = x + y + z
    ders = x1.runreverse()
    assert(x1.get_val() == 10)
    assert(ders == [1, 1, 1]).all()

def test_sub():
    # Scalar
    a = rAd_Var(101)
    b = 1
    f = a - b
    ders = f.runreverse()
    assert(f.get_val() == 100)
    assert(ders == [1])
    # Gradient
    x = rAd_Var(500)
    y = rAd_Var(100)
    z = rAd_Var(-100)
    x1 = x - y - z
    ders = x1.runreverse()
    assert(x1.get_val() == 500)
    assert(ders == [1, -1, -1]).all()

def test_mul():
    # Scalar
    a = rAd_Var(12)
    b = 3
    f = a * b
    g = b * a
    ders_f = f.runreverse()
    ders_g = g.runreverse()
    assert(f.get_val() == g.get_val() == 36)
    assert(ders_f == ders_g) 
    assert(ders_f == [3]).all()
    # Gradient
    x = rAd_Var(3)
    y = rAd_Var(-2)
    z = rAd_Var(3)
    x1 = x * y * z
    ders = x1.runreverse()
    assert(x1.get_val() == -18)
    assert(ders == [-6, 9, -6]).all()

def test_div():
    # Scalar
    a = rAd_Var(20)
    b = 2
    f = a / b
    # g = b / a
    ders_f = f.runreverse()
    # ders_g = g.runreverse()
    assert(f.get_val() == 10)
    # assert(g.get_val() == 0.5)
    # Gradient
    x = rAd_Var(10)
    y = rAd_Var(5)
    x1 = x / y
    ders = x1.runreverse()
    assert(x1.get_val() == 2)

def test_log():
    # Scalar
    a = rAd_Var(5)
    b = a.log()
    ders = b.runreverse()
    assert(b.get_val() == np.log(5))
    assert(ders == .2)
    c = rAd_Var(np.e)
    d = c.log()
    ders = d.runreverse()
    assert(d.get_val() == 1)
    assert(ders == 1/np.e)

def test_pow():
    # Scalar
    a = rAd_Var(2)
    b = 3
    f = a ** b
    ders = f.runreverse()
    assert(f.get_val() == 8)
    assert(ders == [12])
    # Type test.
    try:
        f2 = a ** "NaN"
    except:
        print("test_pow type checking successful!")
    # Rpow.
    c = rAd_Var(4)
    d = 3
    g = d ** c
    ders_g = g.runreverse()
    assert(g.get_val() == 81)
    assert(ders_g == [81*np.log(3)])
    # Gradient
    x = rAd_Var(2)
    y = rAd_Var(2)
    z = rAd_Var(2)
    x1 = x ** y ** z
    ders = x1.runreverse()
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
    ders = f.runreverse()
    assert(f.get_val() == -5)
    assert(ders == -1)

def test_sqrt():
    a = rAd_Var(16)
    f = a.sqrt()
    ders = f.runreverse()
    assert(f.get_val() == 4)
    assert(ders == 1/8)

def test_input():
    try:
        print(rAd_Var('NaN'))
    except TypeError:
        print("Input test 1 passed.")
    try:
        print(rAd_Var(None))
    except TypeError:
        print("Input test 2 passed.")

def test_jacobian():

    def f1(x, y):
        return rAd_Var.cos(x) * (y + 2)

    def f2(x, y):
        return 1 + x ** 2 / (x * y * 3)

    def f3(x, y):
        return 3 * rAd_Var.log(x * 2) + rAd_Var.exp(x / y)

    np.testing.assert_array_almost_equal(rAd_Var.get_jacobian([f1, f2, f3], [1, 2]), [[-3.36588394, 0.54030231],[0.16666667, -0.08333333],[3.82436064, -0.41218032]])

def test_get_val():

    def f1(x, y):
        return rAd_Var.cos(x) * (y + 2)

    def f2(x, y):
        return 1 + x ** 2 / (x * y * 3)

    def f3(x, y):
        return 3 * rAd_Var.log(x * 2) + rAd_Var.exp(x / y)

    return rAd_Var.get_values([f1, f2, f3], [1, 2])

    np.testing.assert_array_almost_equal(rAd_Var.get_jacobian([f1, f2, f3], [1, 2]), [2.16120922, 1.16666667, 3.72816281])

test_exp()
test_add()
test_sub()
test_mul()
test_div()
test_pow()
test_eq()
test_neg()
test_sqrt()
test_input()
test_log()
test_jacobian()
print("All tests passed!")