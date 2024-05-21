import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import datetime
import time
from utils import redirect_log_file, Timer
import argparse
from Dataset import prepare_dataloader
from utils import get_non_pad_mask
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

######################################## Train synthetic data

class Masker(nn.Module):
    def __init__(self, num_types, n_events, n_hidden, n_input):
        super(Masker, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, num_types)

    def forward(self, x):
        z = self.fc2(F.relu(self.fc1(x)))
        m = nn.Softplus()
        z = m(z)
        return z


class A_learner(nn.Module):
    def __init__(self, num_types, n_events, n_hidden, n_input):
        super(A_learner, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, num_types)

    def forward(self, x):
        z = self.fc2(F.relu(self.fc1(x)))
        m = nn.Softplus()
        z = m(z)
        return z


class base_learner(nn.Module):
    def __init__(self, num_types, n_events, n_hidden, n_input):
        super(base_learner, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        z = self.fc2(F.relu(self.fc1(x)))
        m = nn.Softplus()
        z = m(z)
        return z


class PP_model_learning(nn.Module):
    def __init__(self, args):
        super(PP_model_learning, self).__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.relu = nn.ReLU()

        self.time_horizon = args.time_horizon
        self.num_dims = args.num_dims
        self.num_seqs = args.num_seqs

        self.param_path = args.param_path
        self.data_name = args.data_name

        self.num_iters = args.num_iters

        # --- for real data ---
        # self.pattern = torch.zeros((self.num_dims, self.num_dims))
        # self.mu_vector = torch.ones((1, self.num_dims))
        # self.beta_matrix = torch.ones((self.num_dims, self.num_dims))
        # self.A_matrix_truth = torch.zeros((self.num_dims, self.num_dims))
        # self.ground_truth_delta_matrix = torch.zeros((self.num_dims, self.num_dims))
        # ------

        self.pattern, self.mu_vector, self.beta_matrix, self.A_matrix_truth, self.ground_truth_delta_matrix = self.load_ground_truth_params(
            self.param_path, self.data_name)

        self.n_input = 5
        self.n_hidden = 32
        self.masker = Masker(num_types=self.num_dims, n_events=self.num_seqs, n_hidden=self.n_hidden,
                             n_input=self.n_input)
        self.a_learner = A_learner(num_types=self.num_dims, n_events=self.num_seqs, n_hidden=self.n_hidden,
                                   n_input=self.n_input)
        self.mu_learner = base_learner(num_types=self.num_dims, n_events=self.num_seqs, n_hidden=self.n_hidden,
                                       n_input=self.n_input)

        self.lr_A = 1e-3
        self.lr_delta = 1e-3
        self.lr_mu = 1e-3

    def load_ground_truth_params(self, param_path, data_name):
        with open(param_path + data_name + '.txt', "r") as file:
            all_lines = file.readlines()

        target_pattern = "sparsity pattern"
        target_mu = "base vector"
        target_beta = "beta matrix"
        target_A = "A matrix"
        target_delta = "delta matrix"
        pattern_tensor = torch.tensor([0])
        mu_tensor = torch.tensor([0])
        beta_tensor = torch.tensor([0])
        A_tensor = torch.tensor([0])
        delta_tensor = torch.tensor([0])

        for i, line in enumerate(all_lines):
            if target_pattern in line:
                start_line = i + 1
                end_line = start_line + self.num_dims
                selected_lines = all_lines[start_line: end_line]
                pattern_str = ""
                for item in selected_lines:
                    pattern_str += item
                pattern_str = pattern_str.strip().replace("[", "").replace("]", "")
                pattern_rows = pattern_str.split("\n")
                pattern_np = np.array([row.split() for row in pattern_rows], dtype=np.float32)
                pattern_tensor = torch.tensor(pattern_np)
            elif target_mu in line:
                start_line = i + 1
                end_line = start_line + 1
                selected_lines = all_lines[start_line: end_line]
                mu_str = ""
                for item in selected_lines:
                    mu_str += item
                mu_str = mu_str.strip().replace("[", "").replace("]", "")
                mu_rows = mu_str.split("\n")
                mu_np = np.array([row.split() for row in mu_rows], dtype=np.float32)
                mu_tensor = torch.tensor(mu_np)
            elif target_beta in line:
                start_line = i + 1
                end_line = start_line + self.num_dims
                selected_lines = all_lines[start_line: end_line]
                beta_str = ""
                for item in selected_lines:
                    beta_str += item
                beta_str = beta_str.strip().replace("[", "").replace("]", "")
                beta_rows = beta_str.split("\n ")
                beta_np = np.array([row.split() for row in beta_rows], dtype=np.float32)
                beta_tensor = torch.tensor(beta_np)
            elif target_A in line:
                start_line = i + 1
                end_line = start_line + self.num_dims
                selected_lines = all_lines[start_line: end_line]
                A_str = ""
                for item in selected_lines:
                    A_str += item
                A_str = A_str.strip().replace("[", "").replace("]", "")
                A_rows = A_str.split("\n ")
                A_np = np.array([row.split() for row in A_rows], dtype=np.float32)
                A_tensor = torch.tensor(A_np)
            elif target_delta in line:
                start_line = i + 1
                end_line = start_line + self.num_dims
                selected_lines = all_lines[start_line: end_line]
                delta_str = ""
                for item in selected_lines:
                    delta_str += item
                delta_str = delta_str.strip().replace("[", "").replace("]", "")
                delta_rows = delta_str.split("\n ")
                delta_np = np.array([row.split() for row in delta_rows], dtype=np.float32)
                delta_tensor = torch.tensor(delta_np)

        return pattern_tensor, mu_tensor, beta_tensor, A_tensor, delta_tensor

    def intensity(self, delay_effect, mask_matrix, seq, A_matrix, base_train):

        B, D_1, D_2, L = delay_effect.shape
        mask_matrix_1 = mask_matrix.unsqueeze(-3).repeat(1, D_1, 1, 1)

        new_seq_1 = seq.unsqueeze(-3).repeat(1, D_1, 1, 1).to(self.device)

        seq_delay = new_seq_1 + delay_effect.to(self.device)
        seq_delay = mask_matrix_1 * seq_delay

        new_seq = seq.unsqueeze(-2).repeat(1, 1, D_2, 1)
        that_seq = new_seq.unsqueeze(-1).repeat(1, 1, 1, 1, L).to(self.device)
        new_seq_delay = seq_delay.unsqueeze(-2).repeat(1, 1, 1, L, 1).to(self.device)

        k = that_seq - new_seq_delay

        x = torch.sigmoid(1000 * k)
        x = x * mask_matrix_1.unsqueeze(-2).repeat(1, 1, 1, L, 1)

        # A_matrix = self.A_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B, 1, 1, L, L).to(self.device)
        A_matrix = A_matrix.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, L, L).to(self.device)
        beta_matrix = self.beta_matrix.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B, 1, 1, L, L).to(self.device)

        my_need = torch.sum(A_matrix * torch.exp(-beta_matrix * k * x) * x, dim=-1).squeeze()
        my_need = torch.sum(my_need, dim=-2).squeeze()

        intensity = self.mu_vector.unsqueeze(-1)
        intensity = intensity.repeat(B, 1, L).to(self.device)
        intensity[:, -1] = base_train
        intensity = intensity + my_need
        intensity = intensity * mask_matrix

        return intensity

    def compute_integration(self, delay_effect, mask_matrix, seq, A_matrix, base_train):

        B, D_1, D_2, L = delay_effect.shape
        mask_matrix_1 = mask_matrix.unsqueeze(-3).repeat(1, D_1, 1, 1)

        new_seq_1 = seq.unsqueeze(-3).repeat(1, D_1, 1, 1).to(self.device)
        seq_delay = new_seq_1 + delay_effect.to(self.device)
        seq_delay = mask_matrix_1 * seq_delay

        k = torch.zeros([B, self.num_dims, self.num_dims, L]).to(self.device)
        for i in range(B):
            time_horizon = 0
            for val in seq[i, :, -1]:
                if 999 > val > time_horizon:
                    time_horizon = val
            ki = time_horizon - seq_delay[i]
            k[i] = ki

        x = torch.sigmoid(1000 * k)
        x = x * mask_matrix_1

        beta_matrix = self.beta_matrix.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, L).to(self.device)

        my_need = torch.sum(A_matrix / beta_matrix * x * (1 - torch.exp(-beta_matrix * k * x)),
                            dim=-1).squeeze()
        my_need = torch.sum(my_need, dim=-1).squeeze()

        intensity_int = self.mu_vector
        intensity_int = intensity_int.repeat(B, 1).to(self.device)
        for b in range(B):
            intensity_int[b][-1] = base_train[b]

        for i in range(B):
            time_horizon = 0
            for val in seq[i, :, -1]:
                if 999 > val > time_horizon:
                    time_horizon = val
            intensity_int[i] = intensity_int[i] * time_horizon

        intensity_int = intensity_int + my_need

        return intensity_int

    def log_likelihood(self, data_time):
        '''
        Here 'data' is one sequence for multiple dimensions. (e.g., one sequence for one patient)
        '''

        B, num_dims, L = data_time.shape

        # obtain the delay_effect
        x = torch.normal(2, 1, size=(B, self.n_input)).to(self.device)
        delta_matrix_zeros = torch.zeros((B, self.num_dims, self.num_dims)).to(self.device)
        delta_matrix_train = self.masker(x)  # [self.batch_size, self.num_dims], only the last target dim
        for b in range(B):
            delta_matrix_zeros[b][-1, :] = delta_matrix_train[b]  # the last (target)
        delay_effect = delta_matrix_zeros.unsqueeze(-1).repeat(1, 1, 1, L)

        B, D_1, D_2, L = delay_effect.shape

        x = torch.normal(2, 1, size=(B, self.n_input)).to(self.device)  # noise
        A_matrix_zeros = torch.zeros((B, self.num_dims, self.num_dims)).to(self.device)
        A_matrix_train = self.a_learner(x)
        for b in range(B):
            A_matrix_zeros[b][-1, :] = A_matrix_train[b]
        A_matrix = A_matrix_zeros.unsqueeze(-1).repeat(1, 1, 1, L)

        x = torch.normal(2, 1, size=(B, self.n_input)).to(self.device)  # noise
        base_train = self.mu_learner(x)

        # compute log_likelihood
        # get padding
        mask_matrix = get_non_pad_mask(data_time).to(self.device)
        intensity = self.intensity(delay_effect, mask_matrix, data_time, A_matrix_zeros, base_train)
        event_ll = torch.sum(torch.log(intensity + 1e-12) * mask_matrix, dim=-1).squeeze()
        intensity_int = self.compute_integration(delay_effect, mask_matrix, data_time, A_matrix, base_train)

        log_likelihood = event_ll - intensity_int
        log_likelihood = torch.sum(log_likelihood, dim=-1).squeeze()

        return -log_likelihood, delta_matrix_train, A_matrix_train, base_train, event_ll  # truth delay


    def optim_log_likelihood(self, data):
        # Define optimizers
        masker_params = [
            {'params': model.masker.parameters(), 'lr': self.lr_delta}
        ]
        a_learner_params = [
            {'params': model.a_learner.parameters(), 'lr': self.lr_A}
        ]
        mu_learner_params = [
            {'params': model.mu_learner.parameters(), 'lr': self.lr_mu}
        ]
        optimizer_masker = torch.optim.Adam(masker_params, betas=(0.9, 0.999), eps=1e-05)
        optimizer_a_learner = torch.optim.Adam(a_learner_params, betas=(0.9, 0.999), eps=1e-05)
        optimizer_mu_learner = torch.optim.Adam(mu_learner_params, betas=(0.9, 0.999), eps=1e-05)

        scheduler_a_learner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a_learner, T_max=20, eta_min=1e-4)
        scheduler_mask_learner = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_masker, T_max=20, eta_min=1e-4)

        loss_list = []
        event_ll_list = []
        for iter in range(self.num_iters):
            print('----- start the {}-th iteration -----'.format(iter))

            loss_list_batch = []
            event_ll_list_batch = []
            starting_time = time.time()
            delta_matrix_train_all = []
            A_matrix_train_all = []
            mu_vector_train_all = []
            for i, data_time in enumerate(data):
                loss, delta_matrix_train, A_matrix_train, base_train, event_ll = self.log_likelihood(data_time)
                delta_matrix_train_all.extend(delta_matrix_train.cpu().detach().numpy())
                A_matrix_train_all.extend(A_matrix_train.cpu().detach().numpy())
                mu_vector_train_all.extend(base_train.cpu().detach().numpy())
                loss = torch.sum(loss)
                optimizer_a_learner.zero_grad()
                if iter <= 100:
                    optimizer_masker.zero_grad()
                    optimizer_mu_learner.zero_grad()

                loss.backward()

                optimizer_a_learner.step()
                if iter <= 100:
                    optimizer_masker.step()
                    optimizer_mu_learner.step()

                scheduler_a_learner.step()
                scheduler_mask_learner.step()

                loss_list_batch.append(loss.item() / self.batch_size)
                event_ll = torch.sum(event_ll)
                event_ll_list_batch.append(event_ll.item() / self.batch_size)

            ending_time = time.time()
            loss_avg_iter = sum(loss_list_batch) / len(loss_list_batch)
            print('The {}-th iteration, loss = {}'.format(iter, loss_avg_iter))
            loss_list.append(loss_avg_iter)
            event_ll_iter = sum(event_ll_list_batch) / len(event_ll_list_batch)
            print('event_ll = {}'.format(event_ll_iter))
            event_ll_list.append(event_ll_iter)
            print('time cost = {}s'.format(ending_time - starting_time))

            delta_matrix_learned_last_iter = [row for row in delta_matrix_train_all]
            delta_matrix_learned_last_iter = np.transpose(delta_matrix_learned_last_iter)
            learned_delta_means = []
            learned_delta_stds = []
            learned_delta_weights = []
            for d in range(args.num_dims):
                gm = GaussianMixture(n_components=3, random_state=0).fit(delta_matrix_learned_last_iter[d].reshape(-1, 1))
                learned_delta_means.append(gm.means_)
                learned_delta_stds.append(gm.covariances_)
                learned_delta_weights.append(gm.weights_)
            print('----- delta_matrix_learned -----')
            print("mean:", learned_delta_means)
            print("stds:", learned_delta_stds)
            print("weights:", learned_delta_weights)

            A_matrix_learned_last_iter = [row for row in A_matrix_train_all]
            A_matrix_learned_last_iter = np.transpose(A_matrix_learned_last_iter)
            learned_A_means = []
            learned_A_stds = []
            learned_A_weights = []
            for d in range(args.num_dims):
                gm = GaussianMixture(n_components=3, random_state=0).fit(A_matrix_learned_last_iter[d].reshape(-1, 1))
                learned_A_means.append(gm.means_)
                learned_A_stds.append(gm.covariances_)
                learned_A_weights.append(gm.weights_)
            print('----- A_matrix_learned -----')
            print("mean:", learned_A_means)
            print("stds:", learned_A_stds)
            print("weights:", learned_A_weights)


            gm = GaussianMixture(n_components=3, random_state=0).fit(mu_vector_train_all)
            print('----- mu_vector_learned -----')
            print("mean:", gm.means_)
            print("stds:", gm.covariances_)
            print("weights:", gm.weights_)

        return A_matrix_train_all, delta_matrix_train_all, mu_vector_train_all, loss_list, event_ll_list

    def optim(self, data, old_data):

        self.print_info()
        A_matrix_learned_all, delta_matrix_learned_all, mu_vector_learned_all, loss_list, event_ll_list = self.optim_log_likelihood(data)
        return A_matrix_learned_all, delta_matrix_learned_all, mu_vector_learned_all

    ##### Helper Functions
    def print_info(self):
        print("---------- key model information ----------")
        for valuename, value in vars(self).items():
            if isinstance(value, float) or isinstance(value, int) or isinstance(value, list):
                print("{}={}".format(valuename, value))
        print("--------------------", flush=1)


##### main #####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-time_horizon', help='time horizon', type=int, default=20)
    parser.add_argument('-num_seqs', help='total number of sequences', type=int, default=4000)
    parser.add_argument('-num_dims', help='total number of dimension', type=int, default=5)
    parser.add_argument('-num_iters', help='total number of iterations for learning', type=int, default=100)
    parser.add_argument('-param_path', help='path for ground truth parameters', type=str, default='./log/out/')
    parser.add_argument('-data_name', help='name of dataset', type=str, default='gmm_dataset_20T_5dims_6000seqs')
    parser.add_argument('-dataset', type=str, default='./Synthetic_Data/gmm_dataset_20T_5dims_4000seqs.npy')
    # parser.add_argument('-dataset', type=str, default='./Real_Data/Covid_Policy_Tracker/US_dataset_19dims_29seqs_newT0.npy')
    # parser.add_argument('-dataset', type=str, default='./Real_Data/simple_Processed_MIMIC_IV_Data/simple_processed_mimic_iv_data_impV.npy')
    parser.add_argument('-batch_size', type=int, default=32)
    args = parser.parse_args()
    args.device = torch.device('cuda:1')

    # load synthetic dataset
    data = prepare_dataloader(args)
    old_data = np.load(args.dataset, allow_pickle=True).item()

    ##### main #####
    print("Start time is", datetime.datetime.now(), flush=1)
    with Timer("Total running time") as t:
        redirect_log_file('{}T_{}dims_{}seqs_{}iters_hyper_gpu_gmm_final_mu_3.txt'.format(args.time_horizon, args.num_dims, args.num_seqs, args.num_iters))
        # redirect_log_file('us_batch8.txt')
        model = PP_model_learning(args).to(args.device)
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(num_params)
        A_matrix_learned_all, delta_matrix_learned_all, mu_vector_learned_all = model.optim(data, old_data)
        ##### learned parameters
        print('********************')
        print('----- delta_matrix_ground_truth -----')
        print(model.ground_truth_delta_matrix)
        delta_matrix_learned_last_iter = [row for row in delta_matrix_learned_all]
        delta_matrix_learned_last_iter = np.transpose(delta_matrix_learned_last_iter)
        learned_delta_means = []
        learned_delta_stds = []
        learned_delta_weights = []
        for d in range(args.num_dims):
            gm = GaussianMixture(n_components=3, random_state=0).fit(delta_matrix_learned_last_iter[d].reshape(-1, 1))
            learned_delta_means.append(gm.means_)
            learned_delta_stds.append(gm.covariances_)
            learned_delta_weights.append(gm.weights_)
        print('----- delta_matrix_learned -----')
        print("mean:", learned_delta_means)
        print("stds:", learned_delta_stds)
        print("weights:", learned_delta_weights)

        print('----- A_matrix_ground_truth -----')
        print(model.A_matrix_truth)
        A_matrix_learned_last_iter = [row for row in A_matrix_learned_all]
        A_matrix_learned_last_iter = np.transpose(A_matrix_learned_last_iter)
        learned_A_means = []
        learned_A_stds = []
        learned_A_weights = []
        for d in range(args.num_dims):
            gm = GaussianMixture(n_components=3, random_state=0).fit(A_matrix_learned_last_iter[d].reshape(-1, 1))
            learned_A_means.append(gm.means_)
            learned_A_stds.append(gm.covariances_)
            learned_A_weights.append(gm.weights_)
        print('----- A_matrix_learned -----')
        print("mean:", learned_A_means)
        print("stds:", learned_A_stds)
        print("weights:", learned_A_weights)

        print('----- mu_vector_ground_truth -----')
        print(model.mu_vector)
        # params = norm.fit(mu_vector_learned_all)
        # print('----- mu_vector_learned -----')
        # print("mean:", params[0])
        # print("stds:", params[1])
        gm = GaussianMixture(n_components=3, random_state=0).fit(mu_vector_learned_all)
        print('----- mu_vector_learned -----')
        print("mean:", gm.means_)
        print("stds:", gm.covariances_)
        print("weights:", gm.weights_)

    print("Exit time is", datetime.datetime.now(), flush=1)
