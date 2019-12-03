import numpy as np
import numbers as numbers

class rAd_Var():
    def __init__(self, val, ders=None, original = True):
        self._val = np.array([val]).reshape(-1,)
        self._ders = ders
        self.parents = []
        self.children = []
        self.original = original

    def __str__(self):
        return f'Reverse Autodiff Object with value {self._val} and gradient {self.gradient()}'

    def __add__(self, other):
        try:
            rad_object = rAd_Var(self._val + other._val, original=False)
            self.children.append((rad_object, np.ones(len(self._val))))
            other.children.append((rad_object, np.ones(len(self._val))))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val + other, original=False)
            self.children.append((rad_object, np.ones(len(self._val))))
            rad_object.parents = [self]
            return rad_object

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            rad_object = rAd_Var(self._val - other._val, original=False)
            self.children.append((rad_object, np.ones(len(self._val))))
            other.children.append((rad_object, -1 * np.ones(len(self._val))))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val - other, original=False)
            self.children.append((rad_object, np.ones(len(self._val))))
            rad_object.parents = [self]
            return rad_object

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        try:
            rad_object = rAd_Var(self._val * other._val, original=False)
            self.children.append((rad_object, other._val))
            other.children.append((rad_object, self._val))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val * other, original=False)
            self.children.append((rad_object, np.array([other] * len(self._val))))
            rad_object.parents = [self]
            return rad_object

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            rad_object = rAd_Var(self.val / other.val, original=False)
            self.children.append((rad_object, 1 / other._val))
            other.children.append((rad_object, -self._val*(other._val**(-2))))
            return rad_object
        except AttributeError:
            ad = rAd_Var(self.val / other, original=False)
            self.children.append((1/other, ad))
            rad_object.parents = [self]
            return ad

    def __rtruediv__(self, other):
        return other * self**(-1)

    def __pow__(self, other):
        try:
            rad_object = rAd_Var(self._val ** other.val, original=False)
            self.children.append((rad_object, other.val * self._val ** (other.val - 1)))
            other.children.append((rad_object, self._val**other.val*np.log(self._val)))
            rad_object.parents = [self, other]
            return rad_object
        except AttributeError:
            rad_object = rAd_Var(self._val ** other, original=False)
            self.children.append((rad_object, other * self._val ** (other - 1)))
            rad_object.parents = [self]
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
            for var, weight in self.children:
                self._ders += var.gradient() * weight
        return self._ders

    def get_originals(self):
        ancestorlist = []
        if self.parents:
            for parent in self.parents:
                ancestorlist.append(parent)
                ancestorlist += parent.get_originals()
        else:
            pass
        return ancestorlist

    def runreverse(self):
        self._ders = 1.0
        originals = []
        gradient_matrix = np.array([])

        for ancestor in self.get_originals():
            if not ancestor.parents:
                originals.append(ancestor)

        for original in originals:
            gradient_matrix = np.append(gradient_matrix, original.gradient())

        return gradient_matrix

    def exp(self):
        rad_object = rAd_Var(np.exp(self._val))
        self.children.append((rad_object, np.exp(self._val)))
        return rad_object

x = rAd_Var(1)
y = rAd_Var(2)


z = x**2 * y
print(z.runreverse())

# a = z + rAd_Var.exp(z)
# print(x.gradient())
# print(y.gradient())
