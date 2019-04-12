import numpy as np
import random
from math import pow, gamma, sin, pi
from functions import dist_from_zero, rastrigin, booth, Matyas, Rosenbrock, THC, McCormic

class PSOM:
    def __init__(self, num_particles, w_v, function, p1 = 0.1, p2 = 0.1,lbd = 1.5, num = 5):
        self.num_particles = num_particles
        self.generate_values = function.generate_values
        self.particles = self.generate_values(num_particles)
        self.cost = function.eval
        self.shape = function.shape
        self.d = function.d
        self.current_cost = self.evaluate()
        self.min_particle = np.copy(self.particles)
        self.min_cost = np.copy(self.particles[np.argmin(self.current_cost)])
        self.velocities = np.array([np.random.rand(*self.shape)*2*self.d - self.d for i in range(num_particles)])
        self.w_v = w_v
        self.p1 = p1
        self.p2 = p2
        self.lbd = lbd
        self.value = np.power((gamma(1+self.lbd)*sin(pi*self.lbd/2)/(gamma((1+self.lbd)/2)*self.lbd*np.power(2,((self.lbd-1)/2)))),(1/self.lbd))
        self.num = num

    def lambda_func(self):
        return np.array(0.01 * np.random.rand(*self.shape)*self.d * self.value)/np.power(np.abs(np.random.rand(*self.shape)*self.d),1/ self.lbd)

    def evaluate(self):
        current_cost = list()
        for i in self.particles:
            current_cost.append(self.cost(i))
        return np.array(current_cost)

    def optimize(self, num_iterations):
        for iter in range(num_iterations):
            for i in range(len(self.particles)):
                r1 = np.random.rand(*self.shape)
                r2 = np.random.rand(*self.shape)
                self.velocities[i] = (self.w_v * self.velocities[i]) + (self.lambda_func() * r1 * (self.min_particle[i] - self.particles[i])) + (self.lambda_func() * (self.min_cost - self.particles[i]))
                self.particles[i] +=  self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -self.d, self.d)
                self.velocities[i] = np.clip(self.velocities[i], -self.d, self.d)
                if(self.cost(self.particles[i]) < self.cost(self.min_particle[i])):
                    self.min_particle[i] = np.copy(self.particles[i])
            new_values = np.argsort(self.evaluate())[-int(self.p1 * self.num_particles)-int(self.p2 * self.num_particles):-int(self.p2 * self.num_particles)]
            modified_values = np.argsort(self.evaluate())[-int(self.p2 * self.num_particles):]
            if(iter%self.num == 0):
                self.particles[new_values] = self.generate_values(int(self.p1 * self.num_particles))
                self.velocities[new_values] = np.array([np.random.rand(*self.shape)*2*self.d - self.d for i in range(int(self.p1 * self.num_particles))])
                self.particles[modified_values] = np.mean([self.particles[np.argsort(self.evaluate())[:int(self.p1 * self.num_particles)]],np.random.permutation(self.particles[np.argsort(self.evaluate())[:int(self.p1 * self.num_particles)]])],axis = 0)
                self.velocities[modified_values] = np.array([np.random.rand(*self.shape)*2*self.d - self.d for i in range(int(self.p2 * self.num_particles))])
            self.current_cost = self.evaluate()
            self.min_cost = np.copy(self.particles[np.argmin(self.current_cost)])
