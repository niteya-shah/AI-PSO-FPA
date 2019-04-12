import numpy as np

class dist_from_zero:
    def __init__(self, d):
        self.shape = [3]
        self.d  = d

    def eval(self, i):
        return np.linalg.norm(i)

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class rastrigin:
    def __init__(self, d):
        self.shape  = [5]
        self.d = d
        self.A = 10

    def eval(self, i):
        return (self.A*self.shape[0] + np.sum((i*i) - self.A * np.cos(2 * np.pi * i)))

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class booth:
    def __init__(self, d):
        self.shape  = [2]
        self.d = d

    def eval(self, i):
        return (i[0] + 2*i[1] - 7) * (i[0] + 2*i[1] - 7) + (2*i[0] + i[1] - 5) * (2 * i[0] + i[1] - 5)

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class Matyas:
    def __init__(self, d):
        self.shape  = [2]
        self.d = d

    def eval(self, i):
        return 0.26 * (i[0] * i[0] + i[1] * i[1]) - 0.48 * i[0] * i[1]

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class THC:
    def __init__(self, d):
        self.shape  = [2]
        self.d = d

    def eval(self, i):
        return 2 * i[0] * i[0] - 1.05 * np.power(i[0],4) + np.power(i[0],6) + i[0] * i[1] + i[1] * i[1]

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class McCormic:
    def __init__(self, d):
        self.shape  = [2]
        self.d = d

    def eval(self, i):
        return np.sin(i[0] + i[1]) + ((i[0] - i[1]) * (i[0] - i[1])) + ( -1.5 * i[0]) + (2.5 * i[1]) + 1

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)

class Rosenbrock:
    def __init__(self, d):
        self.shape  = [10]
        self.d = d

    def eval(self, i):
        i1 = np.copy(i)[:-1]
        i2 = np.copy(i)[1:]
        return np.sum(100 * (i2 - i1 * i1) * (i2 - i1 * i1) + (1 - i1) * (1 - i1))

    def generate_values(self, num_values):
        values = list()
        for i in range(num_values):
            values.append(np.random.rand(*self.shape)*2*self.d - self.d)
        return np.array(values)
