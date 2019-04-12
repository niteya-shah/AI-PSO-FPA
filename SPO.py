import numpy as np
import random
from functions import dist_from_zero, rastrigin, booth, Matyas, Rosenbrock, THC, McCormic

class PSO:
    def __init__(self, num_particles, w_v, w_lm, w_gm, function):
        self.num_particles = num_particles
        self.particles = function.generate_values(num_particles)
        self.cost = function.eval
        self.shape = function.shape
        self.d = function.d
        self.current_cost = self.evaluate()
        self.min_particle = np.copy(self.particles)
        self.min_cost = np.copy(self.particles[np.argmin(self.current_cost)])
        self.velocities = np.array([np.random.rand(*self.shape)*2*self.d - self.d for i in range(num_particles)])
        self.w_v = w_v
        self.w_lm = w_lm
        self.w_gm = w_gm

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
                self.velocities[i] = (self.w_v * self.velocities[i]) + (self.w_lm * r1 * (self.min_particle[i] - self.particles[i])) + (self.w_gm * r2 * (self.min_cost - self.particles[i]))
                self.particles[i] +=  self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], -self.d, self.d)
                self.velocities[i] = np.clip(self.velocities[i], -self.d, self.d)
                if(self.cost(self.particles[i]) < self.cost(self.min_particle[i])):
                    self.min_particle[i] = np.copy(self.particles[i])
                    if(self.cost(self.particles[i]) < self.cost(self.min_cost)):
                        self.min_cost = np.copy(self.particles[i])
