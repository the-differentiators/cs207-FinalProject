#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
sys.path.append('../')

from src.ReverseAD import rAd_Var as Ad_Var

def test_exp():
    # Scalar
    a = Ad_Var(2)
    f = a.exp()
    ders = f.runreverse()
    assert(f.get_val() == np.exp(2))
    assert(ders == [np.exp(2)])
    # Gradient
    x = Ad_Var(3)
    y = Ad_Var(-4)
    x1 = Ad_Var.exp(x + y)
    ders = x1.runreverse()
    assert(x1.get_val() == np.exp(-1))
    assert(ders == [np.exp(-1), np.exp(-1)]).all()

def test_add():
    # Scalar
    a = Ad_Var(99)
    f = a + 1
    ders = f.runreverse()
    assert(f.get_val() == 100)
    assert(ders == [1])
    # Gradient
    x = Ad_Var(4)
    y = Ad_Var(8)
    z = Ad_Var(-2)
    x1 = x + y + z
    ders = x1.runreverse()
    assert(x1.get_val() == 10)
    assert(ders == [1, 1, 1]).all()

def test_sub():
    # Scalar
    a = Ad_Var(101)
    b = 1
    f = a - b
    ders = f.runreverse()
    assert(f.get_val() == 100)
    assert(ders == [1])
    # Gradient
    x = Ad_Var(500)
    y = Ad_Var(100)
    z = Ad_Var(-100)
    x1 = x - y - z
    ders = x1.runreverse()
    print(ders)
    assert(x1.get_val() == 500)
    assert(ders == [1, -1, -1]).all()
 
test_exp()
test_add()
test_sub()
print("All tests passed!")