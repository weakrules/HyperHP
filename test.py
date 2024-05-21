import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
from utils import redirect_log_file, Timer
import matplotlib.pyplot as plt
import argparse
from Dataset import prepare_dataloader
from utils import get_non_pad_mask
import os
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Prediction(nn.Module):
    def __init__(self, args):
        super(Prediction, self).__init__()
        self.time_horizon = args.time_horizon
        self.num_dims = args.num_dims
        self.num_seqs = args.num_seqs
        self.param_path = args.param_path
        self.data_name = args.data_name
        self.A_learned = torch.Tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0.12, 0.08, 0.21, 0.01, 0.001]
        ])
        self.delta_matrix_learned = torch.Tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3.0, 2.5, 4.0, 1.2, 0.3]

        ])

        self.beta_matrix = torch.ones((self.num_dims, self.num_dims))

        self.mu_vector = torch.Tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

    def intensity(self, cur_time, dim_idx, delay_effect, data):
        intensity = self.mu_vector[:, dim_idx]
        if dim_idx == self.num_dims - 1:
            return intensity
        for neighbor_idx in range(self.num_dims):
            neighbor_history = torch.Tensor(data[neighbor_idx])
            delay = delay_effect[neighbor_idx][dim_idx, :]

            k = cur_time - delay - neighbor_history
            x = torch.sigmoid(1000 * k)
            intensity = intensity + self.A_learned[dim_idx, neighbor_idx] * \
                        torch.sum(
                            torch.exp(
                                -self.beta_matrix[dim_idx, neighbor_idx] * k * x
                            ) * x
                        )  # to avoid the explosion of exp(): the calculated intensity becomes nan
        return intensity

    def intensity_all(self, cur_time, delta_matrix_true, data, delay_effect):
        intensity_all = 0
        for dim_idx in range(self.num_dims):
            intensity_all += self.intensity(cur_time, dim_idx, delay_effect, data)
        return intensity_all

    def predict_next_event_time_pdf(self, t_j, t, delta_matrix_true, data, delay_effect):
        intensity_at_t = self.intensity_all(t, delta_matrix_true, data, delay_effect)
        integral = scipy.integrate.quad(lambda tau: self.intensity_all(tau, delta_matrix_true, data, delay_effect), t_j, t)[0]
        pdf = intensity_at_t * np.exp(-integral)
        return pdf

    def predict(self):

        predicted_time_list = []
        true_time_list = []
        absolute_error_list = []
        A_matrix_true = self.A_learned
        delta_matrix_true = self.delta_matrix_learned

        # iterate over samples
        # integral_resolution = 0.5
        rmse_all = 0
        accu_type = 0
        count_seq = 0
        for id in list(old_data.keys()):
            print('----- start predicting the {}-th patient -----'.format(id))
            # obtain the delay_effect
            delay_effect = {}
            for i in range(self.num_dims):
                delay_effect[i] = torch.unsqueeze(delta_matrix_true[:, i], dim=1).repeat(1, len(old_data[id][i]))

            # start_time = time.time()
            # t_j = old_data[id][4][-2]  # Time of the last event
            # print("last time event", t_j)
            # print(id)
            # for d in range(self.num_dims):
            #     print("dim", d)
            #     print(old_data[id][d][-2])
            #     print(old_data[id][d][-1])
            #
            # t_grad_1 = []
            # for t in np.arange(t_j, 50, integral_resolution):
            #     t_grad_2 = [torch.Tensor([0.0])]
            #     for tau in np.arange(t_j, t, integral_resolution / 2):
            #         for dim_idx in range(self.num_dims):
            #             t_grad_2.append(self.intensity(tau, dim_idx, delay_effect, old_data[id]))
            #     term_exp = torch.exp(-torch.sum(torch.stack(t_grad_2)) * integral_resolution)
            #     for dim_idx in range(self.num_dims):
            #         t_grad_1.append(
            #             t * self.intensity(t, dim_idx, delay_effect, old_data[id]) * term_exp)
            # next_event_time = torch.sum(torch.stack(t_grad_1)) * integral_resolution
            # predicted_time_list.append(next_event_time)
            # print(next_event_time)

            ones = []
            for item in old_data[id]:
                for k in old_data[id][item]:
                    if k != 999:
                        ones.append([item, k])  # [type, time]
            ones = sorted(ones, key=lambda x: x[1])
            diff_predict_true = 0
            correct_type = 0

            for m in range(len(ones) - 2, len(ones) - 1):  # predict next event time and type for all data
                t_j = ones[m][1]  # Time of the last event
                true_t = ones[m + 1][1]
                true_type = ones[m + 1][0]
                # t_j = old_data[id][4][-2]

                integral_func = lambda t: t * self.predict_next_event_time_pdf(t_j, t, delta_matrix_true, old_data[id],
                                                                               delay_effect)
                next_event_time = scipy.integrate.quad(integral_func, t_j, 20)[0]
                predicted_time_list.append(next_event_time)
                print("predict time: ", next_event_time, "true time: ", true_t)
                diff_predict_true += (next_event_time - true_t) ** 2
                true_time_list.append(true_t)

                intensities = []
                for dim_idx in range(self.num_dims):
                    intensities.append(self.intensity(next_event_time, dim_idx, delay_effect, old_data[id]).numpy())
                next_event_type = np.argmax(intensities)
                print("predict type: ", next_event_type, "true type: ", true_type)
                if next_event_type == true_type:
                    correct_type += 1

            # rmse_one = np.sqrt(diff_predict_true / len(ones))
            rmse_one = np.sqrt(diff_predict_true)
            print("rmse for one sample: ", rmse_one, "\n")
            rmse_all += rmse_one
            predict_accu_one = correct_type / len(ones)
            print("predict accu for one sample: ", predict_accu_one, "\n")
            accu_type += predict_accu_one
            count_seq += 1

        return predicted_time_list, true_time_list, rmse_all/count_seq, accu_type/count_seq


##### main #####
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-time_horizon', help='time horizon', type=int, default=20)
    parser.add_argument('-num_seqs', help='total number of sequences', type=int, default=100)
    parser.add_argument('-num_dims', help='total number of dimension', type=int, default=5)
    parser.add_argument('-num_iters', help='total number of iterations for learning', type=int, default=100)
    parser.add_argument('-param_path', help='path for ground truth parameters', type=str, default='./log/out/')
    parser.add_argument('-data_name', help='name of dataset', type=str, default='gmm_dataset_20T_5dims_6000')
    parser.add_argument('-dataset', type=str, default='./Synthetic_Data/test_gmm_dataset_20T_5dims_100seqs.npy')
    # parser.add_argument('-dataset', type=str, default='./Real_Data/simple_Processed_MIMIC_IV_Data/test.npy')
    # parser.add_argument('-dataset', type=str, default='./Real_Data/Covid_Policy_Tracker/China/dataset_17dims_20seqs_newT0.npy')
    parser.add_argument('-batch_size', type=int, default=32)
    args = parser.parse_args()
    args.device = torch.device('cpu')

    # load synthetic dataset
    new_data = prepare_dataloader(args)
    old_data = np.load(args.dataset, allow_pickle=True).item()

    ##### main #####
    print("Start time is", datetime.datetime.now(), flush=1)
    with Timer("Total running time") as t:
        # redirect_log_file('{}T_{}dims_{}seqs_{}iters_test_mle.txt'.format(args.time_horizon, args.num_dims, args.num_seqs, args.num_iters))
        # redirect_log_file('test_gmm_syn.txt')
        predicted_time_list, true_time_list, rmse_all, accu_type = Prediction(args).predict()
        print("rmse_all", rmse_all)
        print("accu_type_all", accu_type)
    print("Exit time is", datetime.datetime.now(), flush=1)
