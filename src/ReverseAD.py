import numpy as np
import numbers as numbers

class rAd_Var():
    def __init__(self, val, ders=None):
        self._val = np.array([val]).reshape(-1,)
        self._ders = ders
        self.dependents = []

    def __str__(self):
        return f'Reverse Autodiff Object with value {self._val} and gradient {self.gradient()})'

    def __add__(self, other):
        try:
            rad_object = rAd_Var(self._val + other._val)
            self.dependents.append((rad_object, np.ones(len(self._val))))
            other.dependents.append((rad_object, np.ones(len(self._val))))
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self.val + other)
            self.dependents.append((rad_object, np.ones(len(self._val))))
            return rad_object

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            rad_object = rAd_Var(self._val - other._val)
            self.dependents.append((rad_object, np.ones(len(self._val))))
            other.dependents.append((rad_object, -1 * np.ones(len(self._val))))
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self.val - other)
            self.dependents.append((rad_object, np.ones(len(self._val))))
            return rad_object

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        try:
            rad_object = rAd_Var(self._val * other._val)
            self.dependents.append((rad_object, other._val))
            other.dependents.append((rad_object, self._val))
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val * other)
            self.dependents.append((rad_object, np.array([other] * len(self._val))))
            return rad_object

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            rad_object = rAd_Var(self.val / other.val)
            self.dependents.append((rad_object, 1 / other._val))
            other.dependents.append((rad_object, -self._val*(other._val**(-2))))
            return rad_object
        except AttributeError:
            ad = rAD(self.val / other)
            self.children.append((1/other, ad))
            return ad

    def __rtruediv__(self, other):
        return other * self**(-1)

    def __pow__(self, other):
        try:
            rad_object = rAd_Var(self.val ** other.val)
            self.dependents.append((rad_object, other.val*self.val**(other.val-1)))
            other.dependents.append((rad_object, self.val**other.val*np.log(self.val)))
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self.val ** other)
            self.dependents.append((rad_object, other*self.val**(other-1)))
            return rad_object

    def __eq__(self, other):
        if self._val == other._val and self.gradient() == other.gradient():
            return True
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def gradient(self):
        if self._ders is None:
            self._ders = 0
            for var, weight in self.dependents:
                self._ders += var.gradient() * weight
        return self._ders

    def exp(self):
        rad_object = rAd_Var(np.exp(self._val))
        self.dependents.append((rad_object, np.exp(self._val)))
        return rad_object

x = rAd_Var([1, 2])
y = rAd_Var([2, 3])

z = x * y
a = z + rAd_Var.exp(z)
a._ders = 1.0

print(x.gradient())
print(3 + 3*np.exp(6))