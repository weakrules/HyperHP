import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import time
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from numpy import random
from utils import redirect_log_file, Timer
import argparse
from numpy.random import normal, poisson

class PP_generator():
    def __init__(self, time_horizon=20, num_dims=2, num_seqs=2000):
        self.time_horizon = time_horizon
        self.num_dims = num_dims
        self.num_seqs = num_seqs

        self.base_range = [0.10, 0.15]
        self.A_range = [0.10, 0.20]
        self.beta_range = [1.0, 1.0]  # fix = 1
        self.delta_range = [3.0, 3.8]

        self.n_components = 3

        self.weights_A = [0.5, 0.2, 0.3]
        self.A_means = [[0.4, 0.3, 0.35], [0.55, 0.6, 0.5], [0.0, 0.0, 0.0]]
        self.A_stds = [[0.1, 0.05, 0.06], [0.03, 0.04, 0.03], [0.0, 0.0, 0.0]]

        # self.A_means = [[0.4, 0.3, 0.35], [0.5, 0.3, 0.1], [0.2, 0.4, 0.6], [0.3, 0.33, 0.35], [0.0, 0.0, 0.0]]
        # self.A_stds = [[0.1, 0.05, 0.06], [0.12, 0.08, 0.01], [0.05, 0.07, 0.1], [0.03, 0.04, 0.03], [0.0, 0.0, 0.0]]
        # self.A_means = [[0.4, 0.3, 0.35],
        #                 [0.5, 0.3, 0.1],
        #                 [0.2, 0.4, 0.6],
        #                 [0.3, 0.33, 0.35],
        #                 [0.4, 0.3, 0.35],
        #                 [0.5, 0.3, 0.1],
        #                 [0.2, 0.4, 0.6],
        #                 [0.25, 0.35, 0.30],
        #                 [0.2, 0.40, 0.15],
        #                 [0.0, 0.0, 0.0]]
        # self.A_stds = [[0.1, 0.05, 0.06],
        #                [0.12, 0.08, 0.01],
        #                [0.05, 0.07, 0.1],
        #                [0.03, 0.04, 0.03],
        #                [0.12, 0.08, 0.01],
        #                [0.05, 0.07, 0.1],
        #                [0.03, 0.04, 0.03],
        #                [0.05, 0.07, 0.1],
        #                [0.03, 0.04, 0.03],
        #                [0.0, 0.0, 0.0]]

        # three components
        self.weights_delay = [0.2, 0.5, 0.3]
        self.delta_means = [[3.0, 4.0, 5.0], [2.0, 2.5, 1.5], [0.0, 0.0, 5.0]]
        self.delta_stds = [[0.3, 0.2, 0.4], [0.45, 0.35, 0.5], [0.0, 0.0, 0.0]]

        # self.delta_means = [[3.0, 4.0, 5.0], [2.0, 2.5, 3.0], [4.0, 4.1, 4.2], [1.0, 1.5, 1.3], [0.0, 0.0, 5.0]]
        # self.delta_stds = [[0.4, 0.6, 0.8], [0.3, 0.2, 0.4], [0.45, 0.35, 0.5], [0.15, 0.2, 0.25], [0.0, 0.0, 0.0]]

        # self.delta_means = [[3.0, 4.0, 5.0],
        #                     [2.0, 2.5, 3.0],
        #                     [4.0, 3.9, 3.8],
        #                     [5.5, 6.0, 6.5],
        #                     [5.0, 4.0, 3.0],
        #                     [2.0, 6.5, 3.0],
        #                     [4.4, 4.5, 5.0],
        #                     [1.8, 2.0, 1.5],
        #                     [6.0, 7.0, 8.0],
        #                     [0.0, 0.0, 0.0]]
        # self.delta_stds = [[0.4, 0.6, 0.5],
        #                    [0.3, 0.2, 0.4],
        #                    [0.45, 0.35, 0.5],
        #                    [0.15, 0.2, 0.25],
        #                    [0.4, 0.3, 0.2],
        #                    [0.3, 0.2, 0.4],
        #                    [0.45, 0.35, 0.5],
        #                    [0.5, 0.4, 0.3],
        #                    [0.15, 0.2, 0.25],
        #                    [0.0, 0.0, 0.0]]

        self.weights_mu = [0.4, 0.2, 0.4]
        self.mu_means = [[0.3, 0.2, 0.5]]  # only for target dim
        self.mu_stds = [[0.05, 0.02, 0.08]]  # only for target dim
        
        self.sparsity_pattern, self.base_vector, self.A_matrix, self.beta_matrix, self.delta_matrix = self.ini_params()
        
        self.start_time = time.time()
    
    def ini_params(self):
        
        while True:
            beta_matrix = np.random.uniform(0, 0, (self.num_dims, self.num_dims))
            for dim_i in range(self.num_dims):
                beta = np.random.uniform(self.beta_range[0], self.beta_range[1])
                for dim_j in range(self.num_dims):
                    beta_matrix[dim_i][dim_j] = beta
            beta_matrix = np.around(beta_matrix, 3)

            # row-i, column-j -- the delay effect caused by occurrence of dimension-j on dimension-i

            base_vector = np.array([[1.0, 1.6, 0.0]])
            A_matrix = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]])

            delta_matrix = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]])

            sparsity_pattern = np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 0.0]])

            # base_vector = np.array([[1.0, 1.4, 1.2, 1.6, 0.0]])
            # A_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0]])
            #
            # delta_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0]])
            #
            # sparsity_pattern = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [1.0, 1.0, 1.0, 1.0, 0.0]])

            # base_vector = np.array([[0.8, 0.9, 1.0, 0.6, 1.1, 0.8, 0.5, 1.2, 0.7, 0.0]])
            # A_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            #
            # delta_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            #
            # sparsity_pattern = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
            # check stationary condition
            M = A_matrix / beta_matrix
            a, b = np.linalg.eig(M) # 'a' is the eigenvalue set, 'b' is the eigenvalue vector
            rho = np.max(np.abs(a)) # rho is the spectral radius
            if rho < 1:
                print('stationary condition is satisfied')
                break
            else:
                print('stationary condition is not satisfied')
                
        return sparsity_pattern, base_vector, A_matrix, beta_matrix, delta_matrix
               
    def generate_networked_hawkes(self):
        for k in range(self.num_dims - 1):  # last dim does not have self trigger/self delay
            component = np.random.choice(np.arange(self.n_components), p=self.weights_delay)
            self.delta_matrix[-1, k] = np.random.normal(loc=self.delta_means[k][component],
                                                        scale=self.delta_stds[k][component], size=1)
        self.delta_matrix[self.delta_matrix < 0] = 1e-4  # non-negative

        for k in range(self.num_dims - 1):  # last dim does not have self trigger/self delay
            component = np.random.choice(np.arange(self.n_components), p=self.weights_A)
            self.A_matrix[-1, k] = np.random.normal(loc=self.A_means[k][component],
                                                    scale=self.A_stds[k][component], size=1)
        self.A_matrix[self.A_matrix < 0] = 1e-4  # non-negative

        component = np.random.choice(np.arange(self.n_components), p=self.weights_mu)
        base_targets = np.random.normal(loc=self.mu_means[0][component], scale=self.mu_stds[0][component], size=1)
        self.base_vector[-1][-1] = base_targets
        base_targets[base_targets < 0] = 1e-4  # non-negative
        # initialize sequence
        sequence = {i: [] for i in range(self.num_dims)}
        
        # initialize current intensity
        mu_vector = self.base_vector
        cur_intensity = mu_vector[0, :]

        # record delay effect history for each dimension
        delay_effect_history = {i: [] for i in range(self.num_dims)}

        # record delay effect working time history for each dimension
        delay_effect_working_time = {i: [] for i in range(self.num_dims)}

        # record the event occurred at which dimension, naturally sorted by time ascending
        event_idx_list = []

        cur_t = 0
        while True:
            # I_max should reflect the future; updating the intensity requires going back to history
            # to ensure that I_max is the upper bound before the occurring of the next event
            I_max = np.sum(cur_intensity)
            for i in range(self.num_dims):
                if len(delay_effect_working_time[i]) > 0:
                    for j in range(len(delay_effect_working_time[i])):
                        I_max = I_max + self.A_matrix[i, event_idx_list[j]] * (cur_t < delay_effect_working_time[i][j])

            t_delta = np.random.exponential(1.0 / I_max)
            
            cur_t = cur_t + t_delta
            cur_t = round(cur_t, 3)
            if cur_t > self.time_horizon:
                break

            cur_intensity = self.base_vector[0].copy()

            # go back to history
            for i in range(self.num_dims): # cur_t > delay_effect working time
                if len(delay_effect_working_time[i]) > 0:
                    for j in range(len(delay_effect_working_time[i])):
                        if cur_t > delay_effect_working_time[i][j]:
                            cur_intensity[i] += self.A_matrix[i, event_idx] * \
                                                np.exp((-self.beta_matrix[i, event_idx]) * (
                                                            cur_t - delay_effect_working_time[i][j]))

            if np.random.uniform() <= np.sum(cur_intensity) / I_max:
                # accept, event_idx is the corresponding dimension that the event occurs
                event_idx = np.random.choice(self.num_dims, 1, p=cur_intensity / np.sum(cur_intensity))
                event_idx_list.append(event_idx.item())
                sequence[event_idx[0]].append(cur_t)

                # update current intensity
                for neighbor_idx in range(self.num_dims):
                    # boost alpha_{ij}
                    delta = self.delta_matrix[neighbor_idx, event_idx][0]
                    delay_effect_history[neighbor_idx].append(delta)
                    delay_effect_working_time[neighbor_idx].append(cur_t + delta)
        
        return sequence

    def generate_multiple_sequences(self):
        multiple_sequences = {}
        num = 0
        while num < self.num_seqs:
            starting_t = time.time()
            sequence = self.generate_networked_hawkes()
            save = True
            for j in range(self.num_dims):
                if len(sequence[j]) == 0:
                    save = False
                    print('The {}-th sequence has at least one dimension with all zero'.format(num))
                    break
            if save == False:
                continue
            ending_t = time.time()
            running_t = ending_t - starting_t
            print('----- generate the {}-th sequence using t = {}s -----'.format(num, running_t))
            multiple_sequences[num] = sequence
            num += 1
        
        total_ending_t = time.time()
        total_running_t = total_ending_t - self.start_time
        print('========================================')
        print('===== Totally generate {} sequences using t = {}s ====='.format(self.num_seqs, total_running_t))
        return self.sparsity_pattern, self.base_vector, self.beta_matrix, self.A_means, self.A_stds, self.delta_means, self.delta_stds, self.mu_means, self.mu_stds, multiple_sequences
    

#### main #####
time_horizon = 10
num_dims = 3
num_seqs = 6000
print("Start time is", datetime.datetime.now(), flush=1)
with Timer("Total running time") as t:
    redirect_log_file('gmm_dataset_{}T_{}dims_{}seqs.txt'.format(str(time_horizon), str(num_dims), str(num_seqs)))
    PP_generator = PP_generator(time_horizon, num_dims, num_seqs)
    pattern_true, mu_true, beta_true, A_true_mean, A_true_std, delta_true_mean, delta_true_std, mu_true_mean, mu_true_std, multiple_sequences = PP_generator.generate_multiple_sequences()
    A_true = np.zeros((num_dims, num_dims))
    delta_true = np.zeros((num_dims, num_dims))
    print('----- ground truth sparsity pattern -----')
    print(pattern_true)
    print('----- ground truth base vector -----')
    print(mu_true)
    print('----- ground truth A matrix -----')
    print(A_true)
    print('----- ground truth beta matrix -----')
    print(beta_true)
    print('----- ground truth delta matrix -----')
    print(delta_true)
    print('----- ground truth A mean -----')
    print(A_true_mean)
    print('----- ground truth A std -----')
    print(A_true_std)
    print('----- ground truth delta mean -----')
    print(delta_true_mean)
    print('----- ground truth delta std -----')
    print(delta_true_std)
    print('----- ground truth mu mean -----')
    print(mu_true_mean)
    print('----- ground truth mu std -----')
    print(mu_true_std)

    # save data
    if not os.path.exists("./Synthetic_Data"):
        os.makedirs("./Synthetic_Data")
    path = os.path.join("./Synthetic_Data", 'gmm_dataset_{}T_{}dims_{}seqs.npy'.format(str(time_horizon), str(num_dims), str(num_seqs)))
    np.save(path, multiple_sequences)
    # print(multiple_sequences)

print("Exit time is", datetime.datetime.now(), flush=1)
