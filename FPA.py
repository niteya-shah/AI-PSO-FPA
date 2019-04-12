import numpy as np
import random
from math import pow, gamma, sin, pi
from functions import dist_from_zero, rastrigin, booth, Matyas, Rosenbrock, THC, McCormic

class FPA:
    def __init__(self, num_flowers, p, lbd, function):
        self.num_flowers = num_flowers
        self.p = p
        self.lbd = lbd
        self.flowers = function.generate_values(num_flowers)
        self.cost = function.eval
        self.shape = function.shape
        self.d = function.d
        self.value = np.power((gamma(1+self.lbd)*sin(pi*self.lbd/2)/(gamma((1+self.lbd)/2)*self.lbd*np.power(2,((self.lbd-1)/2)))),(1/self.lbd))
        self.current_cost = self.evaluate()
        self.min_cost = self.flowers[np.argmin(self.current_cost)]

    def evaluate(self):
        current_cost = list()
        for i in self.flowers:
            current_cost.append(self.cost(i))
        return np.array(current_cost)

    def lambda_func(self):
        return np.array(0.01 * np.random.rand(*self.shape)*self.d * self.value)/np.power(np.abs(np.random.rand(*self.shape)*self.d),1/ self.lbd)

    def optimize(self, num_iterations):
        for iter in range(num_iterations):
            new_flowers = list()
            for flower in self.flowers:
                if(np.random.rand() < self.p):
                    flower_dash = flower + self.lambda_func() * (flower - self.min_cost)
                else:
                    i = np.random.randint(self.num_flowers)
                    j = np.random.randint(self.num_flowers)
                    flower_dash = flower + np.random.rand() * (self.flowers[i] - self.flowers[j])
                new_flowers.append(flower_dash)
            new_flowers = np.concatenate((np.array(new_flowers), np.array(self.flowers)))
            self.flowers  = np.array(sorted(new_flowers, key = lambda x: self.cost(x))[0:self.num_flowers])
            self.min_cost = self.flowers[0]
