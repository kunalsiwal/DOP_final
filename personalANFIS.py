import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from sklearn.metrics import mean_squared_error
import random
import copy

class GAANFIS:

    def __init__(self, n_input, n_mf, n_output, population_size=10, max_generations=50, mutation_rate=0.1):
        self.n_input = n_input
        self.n_mf = n_mf
        self.n_output = n_output
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness = []
        self.best_individual = None
        self.best_fitness = np.inf
        self.final_weights=[]

    def initialize_population(self):
        for i in range(self.population_size):
            individual = self.generate_individual()
            self.population.append(individual)

    def generate_individual(self):
        individual = {}
        for j in range(self.n_output):
            input_mfs = []
            for i in range(self.n_input):
                mf_params = []
                for k in range(self.n_mf):
                    a = random.uniform(0, 1)
                    b = random.uniform(0, 1)
                    c = random.uniform(0, 1)
                    mf_params.append([a, b, c])
                input_mfs.append(mf_params)
            individual[j] = input_mfs
        return individual
    
  

    def train(self, X_train, y_train):
        self.initialize_population()
        for generation in range(self.max_generations):
            self.fitness = []
            for individual in self.population:
                fitness = self.evaluate_fitness(individual, X_train, y_train)
                self.fitness.append(fitness)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual
            next_population = []
            for i in range(self.population_size):
                parent1 = self.select_individual()
                parent2 = self.select_individual()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
            self.population = next_population

    def evaluate_fitness(self, individual, X_train, y_train):
        y_pred = []
        for x in X_train:
            outputs = []
            for j in range(self.n_output):
                inputs = []
                for i in range(self.n_input):
                    input_mfs = individual[j][i]
                    
                    mf_outputs = []
                  
                    for a, b, c in input_mfs:
                        
                        mf_output = fuzz.interp_membership(np.array([-1,0,1]), np.array([a, b, c]), x[i])
                        mf_outputs.append(mf_output)
                    
                    inputs.append(mf_outputs)
                inputs = np.array(inputs)
                rule_outputs = np.min(inputs, axis=1)
                rule_outputs = np.prod(rule_outputs)
                outputs.append(rule_outputs)
            y_pred.append(np.sum(outputs))
        fitness = mean_squared_error(y_train, y_pred)
        return fitness

    def select_individual(self):
        fitness_sum = sum(self.fitness)
        fitness_probs = [fitness / fitness_sum for fitness in self.fitness]
        return self.population[np.random.choice(self.population_size, p=fitness_probs)]

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for j in range(self.n_output):
            for i in range(self.n_input):
                if random.random() < 0.5:
                    child[j][i] = copy.deepcopy(parent2[j][i])
        return child

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)
        
        for j in range(self.n_output):
            for i in range(self.n_input):
                if random.random() < self.mutation_rate:
                    input_mfs = individual[j][i]
                    for k in range(self.n_mf):
                        a, b, c = input_mfs[k]
                        if random.random() < 0.5:
                            a += random.uniform(-0.1, 0.1)
                        else:
                            a *= random.uniform(0.9, 1.1)
                        if random.random() < 0.5:
                            b += random.uniform(-0.1, 0.1)
                        else:
                            b *= random.uniform(0.9, 1.1)
                        if random.random() < 0.5:
                            c += random.uniform(-0.1, 0.1)
                        else:
                            c *= random.uniform(0.9, 1.1)
                        input_mfs[k] = [a, b, c]
                    mutated_individual[j][i]=input_mfs
        
        return mutated_individual

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            outputs = []
            for j in range(self.n_output):
                inputs = []
                for i in range(self.n_input):
                    input_mfs = self.best_individual[j][i]
                    mf_outputs = []
                    for a, b, c in input_mfs:
                        mf_output = fuzz.interp_membership(np.array([-1,0, 1]), np.array([a, b, c]), x[i])
                    
                        mf_outputs.append(mf_output)
                    inputs.append(mf_outputs)
                inputs = np.array(inputs)
                print(inputs.shape)
                rule_outputs = np.min(inputs, axis=1)
                rule_outputs = np.prod(rule_outputs)
                outputs.append(rule_outputs)
            normalized_outputs=np.linalg.norm(outputs)
            print(normalized_outputs)
            y_pred.append(np.sum(outputs))
        return y_pred


    
