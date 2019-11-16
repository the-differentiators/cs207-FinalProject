#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:24:08 2019

@author: Ivywang
"""

import pytest
import numpy as np
from AutoDiff import Ad_Var

class Test_functions():

    def test_exp(self):
        x1 = Ad_Var(1,np.array([1, 0]))
        x2 = Ad_Var(1,np.array([0, 1]))
        f = Ad_Var.exp(x1 + x2)
        assert f.get_val() == np.exp(1+1)
        assert (f.get_ders() == [np.exp(1+1), np.exp(1+1)]).all()

    def test_log(self):
        x1 = Ad_Var(1, np.array([1, 0]))
        x2 = Ad_Var(1, np.array([0, 1]))
        f = Ad_Var.log(x1 + x2)
        assert f.get_val() == np.log(1+1)
        assert (f.get_ders() == [1/2, 1/2]).all()

    def test_trig(self):
        x1 = Ad_Var(np.pi/4, np.array([1, 0, 0]))
        x2 = Ad_Var(np.pi/4, np.array([0, 1, 0]))
        x3 = Ad_Var(np.pi/4, np.array([0, 0, 1]))
        f = Ad_Var.sin(x1) + Ad_Var.cos(x2) + Ad_Var.tan(x3)
        assert f.get_val() == 2**0.5 + 1
        assert (f.get_ders() == [np.cos(np.pi/4), -np.sin(np.pi/4), 1/np.cos(np.pi/4)**2]).all()

    def test_inverse_trig(self):
        x1 = Ad_Var(0.1, np.array([1, 0, 0]))
        x2 = Ad_Var(0.2, np.array([0, 1, 0]))
        x3 = Ad_Var(0.3, np.array([0, 0, 1]))
        f = Ad_Var.arcsin(x1) + Ad_Var.arccos(x2) + Ad_Var.arctan(x3)
        assert f.get_val() == 1.7610626216439926
        assert (f.get_ders() == [1.005037815259212, -1.0206207261596576, 0.9174311926605504]).all()

    def test_pow(self):
        x1 = Ad_Var(1, np.array([1, 0]))
        x2 = Ad_Var(2, np.array([0, 1]))
        f = x1**2 + x2**(-3)
        assert f.get_val() == 1.125
        assert (f.get_ders() == [2, -3/16]).all()
        
    def test_div1(self):
        x1 = Ad_Var(1, np.array([1, 0]))
        x2 = Ad_Var(2, np.array([0, 1]))
        f = (x1+1)/x2
        assert f.get_val() == 1
        assert (f.get_ders() == [1/2, -1/2]).all()
    
    def test_div2(self):
        x1 = Ad_Var(1)
        x2 = 2
        f = (x1+1)/x2 + x2/(x1+1)
        assert f.get_val() == 2
        assert f.get_ders() == 0 #1/2 - 1/2
        
    def test_mul1(self):
        x1 = Ad_Var(1, np.array([1, 0]))
        x2 = Ad_Var(2, np.array([0, 1]))
        f = (x1+1)*x2
        assert f.get_val() == 4
        assert (f.get_ders() == [2, 2]).all()
    
    def test_mul2(self):
        x1 = Ad_Var(1)
        x2 = 2
        f = (x1+1)*x2 + x2*(x1+1)
        assert f.get_val() == 8
        assert f.get_ders() == 4 
