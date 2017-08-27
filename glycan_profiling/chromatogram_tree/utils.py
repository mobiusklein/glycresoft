from collections import Counter


class ArithmeticMapping(Counter):
    def __init__(self, base=None, **kwargs):
        if base is not None:
            self.update(base)
        else:
            if kwargs:
                self.update(kwargs)

    def __missing__(self, key):
        return 0

    def __mul__(self, i):
        inst = self.copy()
        for key, value in self.items():
            inst[key] = value * i
        return inst

    def __imul__(self, i):
        for key, value in list(self.items()):
            self[key] = value * i
        return self

    def __div__(self, i):
        inst = self.copy()
        for key, value in self.items():
            inst[key] = value / i
        return inst

    def __idiv__(self, i):
        for key, value in list(self.items()):
            self[key] = value / i
        return self
