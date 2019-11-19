#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:24:08 2019

@author: Ivywang
"""


import numpy as np
#from src.AutoDiff import Ad_Var
import os
path = os.getcwd().replace('test','src')
os.chdir(path)

from AutoDiff import Ad_Var
def test_exp():
    ## scaler
    a = Ad_Var(1,-3)
    b = Ad_Var.exp(a)
    assert b.get_val() == np.exp(1)
    assert b.get_ders() == -3*np.exp(1)
    ## gradient
    x1 = Ad_Var(1,np.array([1, 0]))
    x2 = Ad_Var(1,np.array([0, 1]))
    f = Ad_Var.exp(x1 + x2)
    assert f.get_val() == np.exp(1+1)
    assert (f.get_ders() == [np.exp(1+1), np.exp(1+1)]).all()

def test_log():
    ## scaler
    a = Ad_Var(1,-3)
    b = Ad_Var.log(a)
    assert b.get_val() == np.log(1)
    assert b.get_ders() == -3
    ## gradient
    x1 = Ad_Var(1, np.array([1, 0]))
    x2 = Ad_Var(1, np.array([0, 1]))
    f = Ad_Var.log(x1 + x2)
    assert f.get_val() == np.log(1+1)
    assert (f.get_ders() == [1/2, 1/2]).all()

def test_trig():
    ## scaler
    a = Ad_Var(np.pi/4,-3)
    b = Ad_Var.sin(a)
    c = Ad_Var.cos(a)
    d = Ad_Var.tan(a)
    assert b.get_val() == np.sin(np.pi/4)
    assert b.get_ders() == -3*np.cos(np.pi/4)
    assert c.get_val() == np.cos(np.pi/4)
    assert c.get_ders() == 3*np.sin(np.pi/4)
    assert d.get_val() == np.tan(np.pi/4)
    assert d.get_ders() == -3/np.cos(np.pi/4)**2
    ## gradient
    x1 = Ad_Var(np.pi/4, np.array([1, 0, 0]))
    x2 = Ad_Var(np.pi/4, np.array([0, 1, 0]))
    x3 = Ad_Var(np.pi/4, np.array([0, 0, 1]))
    f = Ad_Var.sin(x1) + Ad_Var.cos(x2) + Ad_Var.tan(x3)
    assert f.get_val() == 2**0.5 + 1
    assert (f.get_ders() == [np.cos(np.pi/4), -np.sin(np.pi/4), 1/np.cos(np.pi/4)**2]).all()

def test_inverse_trig():
    ## scaler
    a = Ad_Var(0.1,-3)
    b = Ad_Var.arcsin(a)
    c = Ad_Var.arccos(a)
    d = Ad_Var.arctan(a)
    assert b.get_val() == np.arcsin(0.1)
    assert b.get_ders() == -3/np.sqrt(1-0.1**2)
    assert c.get_val() == np.arccos(0.1)
    assert c.get_ders() == 3/np.sqrt(1-0.1**2)
    assert d.get_val() == np.arctan(0.1)
    assert d.get_ders() == -3/(1+0.1**2)
    ## gradient
    x1 = Ad_Var(0.1, np.array([1, 0, 0]))
    x2 = Ad_Var(0.2, np.array([0, 1, 0]))
    x3 = Ad_Var(0.3, np.array([0, 0, 1]))
    f = Ad_Var.arcsin(x1) + Ad_Var.arccos(x2) + Ad_Var.arctan(x3)
    assert f.get_val() == 1.7610626216439926
    assert (f.get_ders() == [1.005037815259212, -1.0206207261596576, 0.9174311926605504]).all()

def test_pow():
    ## scalar
    a = Ad_Var(1,-3)
    b = a**2
    assert b.get_val() == 1
    assert b.get_ders() == -6
    ## gradient
    x1 = Ad_Var(1, np.array([1, 0]))
    x2 = Ad_Var(2, np.array([0, 1]))
    f = x1**2 + x2**(-3)
    assert f.get_val() == 1.125
    assert (f.get_ders() == [2, -3/16]).all()
    try:
        a = Ad_Var(1,3)
        b = a**'s'
    except TypeError:
        print("Type Error sucessfully catched - pow")
    
def test_sub1():
    x1 = Ad_Var(1, np.array([1, 0]))
    x2 = Ad_Var(2, np.array([0, 1]))
    f = x2 - x1
    assert f.get_val() == 1
    assert (f.get_ders() == [-1, 1]).all()

def test_sub2():
    x1 = Ad_Var(1)
    x2 = 2
    f = x2 - x1
    assert f.get_val() == 1
    assert f.get_ders() == 1

def test_sub3():
    x1 = Ad_Var(1,[1,0])
    x2 = 2
    f = x2 - x1
    assert f.get_val() == 1
    assert (f.get_ders() == [1, 0]).all()

def test_div1():
    x1 = Ad_Var(1, np.array([1, 0]))
    x2 = Ad_Var(2, np.array([0, 1]))
    f = (x1+1)/x2
    assert f.get_val() == 1
    assert (f.get_ders() == [1/2, -1/2]).all()

def test_div2():
    x1 = Ad_Var(1)
    x2 = 2
    f = (x1+1)/x2 + x2/(x1+1)
    assert f.get_val() == 2
    assert f.get_ders() == 0 #1/2 - 1/2
    
def test_mul1():
    x1 = Ad_Var(1, np.array([1, 0]))
    x2 = Ad_Var(2, np.array([0, 1]))
    f = (x1+1)*x2
    assert f.get_val() == 4
    assert (f.get_ders() == [2, 2]).all()

def test_mul2():
    x1 = Ad_Var(1)
    x2 = 2
    f = (x1+1)*x2 + x2*(x1+1)
    assert f.get_val() == 8
    assert f.get_ders() == 4 

def test_multiple():
    x = Ad_Var(1, np.array([1, 0, 0]))
    y = Ad_Var(2, np.array([0, 1, 0]))
    z = Ad_Var(3, np.array([0, 0, 1]))
    f = np.array([Ad_Var.cos(x)*(y+2), 1 + z**2/(x*y*3), 3*Ad_Var.log(x*2) + Ad_Var.exp(x/z), Ad_Var.arctan(Ad_Var.arcsin(y/4))])
    assert (Ad_Var.get_values(f) == [np.cos(1)*4, 1 + 3**2/6 ,3*np.log(2) + np.exp(1/3), np.arctan(np.arcsin(1/2))]).all()
    assert (Ad_Var.get_jacobian(f, 4, 3) == [[-4*np.sin(1), np.cos(1), 0],[-1.5, -0.75,1],[3+np.exp(1/3)/3,0,-np.exp(1/3)/9],[0, 1/(4*(np.sqrt(3/4))*(np.arcsin(1/2)**2 + 1)),0]]).all()

def test_eq():
    a = Ad_Var(1, -1)
    b = Ad_Var(1, -1)
    print(a)
    assert a == b
    x = Ad_Var(1, np.array([1,0]))
    y = Ad_Var(1, np.array([1,0]))
    print(x)
    assert (x == y).all()

def test_input():
    try:
        a = Ad_Var('s',2)
    except TypeError:
        print("TypeError sucessfully catched - value1")
    try:
        b = Ad_Var([2,3,4],2)
    except TypeError:
        print("TypeError sucessfully catched - value2")
    try:
        c = Ad_Var(2,'s')
    except TypeError:
        print("TypeError sucessfully catched - der1")
    try:
        d = Ad_Var(2,np.array([1,2,'dog']))
    except TypeError:
        print("TypeError sucessfully catched - der2")
        
def test_func():
    x = Ad_Var(1, np.array([1, 0, 0]))
    y = Ad_Var(2, np.array([0, 1, 0]))
    z = 's'
    try:
        f = np.array([Ad_Var.cos(x)*(y+2), 1 + z**2/(x*y*3), 3*Ad_Var.log(x*2) + Ad_Var.exp(x/z), Ad_Var.arctan(Ad_Var.arcsin(y/4))])
        Ad_Var.get_values(f)
    except TypeError:
        print("TypeError sucessfully catched - get_values")
    try:
        f = np.array([Ad_Var.cos(x)*(y+2), 1 + z**2/(x*y*3), 3*Ad_Var.log(x*2) + Ad_Var.exp(x/z), Ad_Var.arctan(Ad_Var.arcsin(y/4))])
        Ad_Var.get_jacobian(f)
    except TypeError:
        print("TypeError sucessfully catched - get_jacobian")
    try:
        f = np.array([Ad_Var.cos(x)*(y+2), 1 + x**2/(x*y*3), 3*Ad_Var.log(x*2) + Ad_Var.exp(x/y), Ad_Var.arctan(Ad_Var.arcsin(y/4))])
        Ad_Var.get_jacobian(f,5,4)
    except ValueError:
        print("ValueError sucessfully catched - get_jacobian")
 
if __name__ == "__main__":
    test_exp()
    test_log()
    test_trig()
    test_inverse_trig()
    test_pow()
    test_sub1()
    test_sub2()
    test_div1()
    test_div2()
    test_mul1()
    test_mul2()
    test_multiple()
    test_eq()
    test_func()
    test_input()
    print("All tests passed!")