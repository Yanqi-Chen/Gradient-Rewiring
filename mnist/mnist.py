import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import torchvision
from torchvision import transforms

from spikingjelly.clock_driven.functional import reset_net, set_monitor

import numpy as np
import os
import sys
import time
import argparse
#from tqdm import tqdm

from model import MnistNet

sys.path.append('..')

from gradrewire import GradRewiring
from deeprewire import DeepRewiring

############## Reproducibility ##############
_seed_ = 2020
np.random.seed(_seed_)
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-penalty', type=float, default=1e-3)
parser.add_argument('-s', '--sparsity', type=float)
parser.add_argument('-gpu', type=str)
parser.add_argument('--dataset-dir', type=str)
parser.add_argument('--dump-dir', type=str)
parser.add_argument('-T', type=int, default=8)
parser.add_argument('-N', '--epoch', type=int, default=512)
parser.add_argument('-soft', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument(
    '-m', '--mode', choices=['deep', 'grad', 'no_prune'], default='no_prune')

# Epoch interval when recording data (firing rate, acc. on test set, etc.) on TEST set
parser.add_argument('-i1', '--interval-test', type=int, default=32)

# Step interval when recording data (loss, acc. on train set) on TRAIN set
parser.add_argument('-i2', '--interval-train', type=int, default=1024)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = args.batch_size
learning_rate = args.learning_rate
dataset_dir = args.dataset_dir
dump_dir = args.dump_dir
T = args.T
penalty = args.penalty
s = args.sparsity
soft = args.soft
test = args.test
no_prune = (args.mode == 'no_prune')
i1 = args.interval_test
i2 = args.interval_train
N = args.epoch


if __name__ == "__main__":

    file_prefix = 'lr-' + np.format_float_scientific(learning_rate, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'

    if not no_prune:
        file_prefix += '-penalty-' + np.format_float_scientific(penalty, exp_digits=1, trim='-')

    if soft:
        file_prefix = 'soft-' + file_prefix

    if s is not None and not no_prune:
        file_prefix += f'-s-{s}'

    file_prefix += '-' + args.mode

    log_dir = os.path.join(dump_dir, 'logs', file_prefix)
    model_dir = os.path.join(dump_dir, 'models', file_prefix)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ])

    train_dataset = torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=transform_train,
            download=True)
    test_dataset = torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=transform_test,
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, 
        num_workers=8, 
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, 
        num_workers=8, 
        pin_memory=True)

    # Load existing model or create a new one
    if os.path.exists(os.path.join(model_dir, 'net.pkl')):
        net = torch.load(os.path.join(model_dir, 'net.pkl'), map_location='cuda')
        print(f'Load existing model, Train steps: {net.train_times}, Epochs: {net.epochs}')
    else:
        net = MnistNet(T=T).cuda()
        print(f'Create new model')

    ttl_cnt = 0.0 # Number of all parameters

    for param in net.parameters():
        ttl_cnt += param.numel()

    ###### TEST MODE ######
    if test:
        with torch.no_grad():
            # Turn on monitor
            set_monitor(net, True)

            # Record total spike times by layer
            spike_times = dict()

            for name, module in net.named_modules():
                if hasattr(module, 'monitor'):
                    spike_times[name] = 0

            test_sum = 0
            correct_sum = 0

            for img, label in test_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                out_spikes_counter = net(img)

                correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()

                for name, module in net.named_modules():
                    if hasattr(module, 'monitor'):
                        # monitor['s'] is a list, each element is of shape [batch_size, ...]
                        spike_times[name] += torch.sum(torch.from_numpy(np.concatenate(module.monitor['s'], axis=0)).cuda(), dim=0)

                reset_net(net)

            test_accuracy = correct_sum / test_sum

############ 1. Firing Rate ###########
            print('Firing Rates:')
            for k, v in spike_times.items():
                rate = (v / (T * len(test_dataset))).flatten().cpu().numpy()

                if no_prune:
                    filename = 'rate-' + k + '-no_prune.npy'  
                else:
                    filename = 'rate-' + k + '-' + np.format_float_scientific(penalty, exp_digits=1, trim='-') + '.npy'

                with open(os.path.join(log_dir, filename), 'wb') as f:
                    np.save(f, rate)

######### 2. Sparsity & Acc. #########
            if no_prune:
                print(f'Test Acc: {test_accuracy * 100:.3f}%')
            else:
                print('Sparsity:')
                zero_cnt = 0.0
                for name, param in net.named_parameters():
                    curr_zero_cnt = (param == 0.0).float().sum()
                    zero_cnt += curr_zero_cnt
                    print(f'{name}: {curr_zero_cnt / param.numel() * 100:.3f}%')
                        
                sparsity_all = zero_cnt / ttl_cnt

                print(f'Test Acc: {test_accuracy * 100:.3f}%, Sparsity: {sparsity_all * 100:.3f}%')

    ###### TRAIN MODE ######
    else:
        # Recover from unexpected breakpoint of training
        if os.path.exists(os.path.join(model_dir, 'net.pkl')):
            if no_prune:
                optimizer_all = Adam(net.parameters(), lr=learning_rate)
                optimizer_checkpoint = torch.load(os.path.join(model_dir, 'optim.pkl'))
                optimizer_all.load_state_dict(optimizer_checkpoint)

            else:
                if args.mode == 'grad':
                    optimizer_all = GradRewiring(net.parameters(), lr=learning_rate, alpha=penalty, s=s)
                elif args.mode == 'deep':
                    optimizer_all = DeepRewiring(net.parameters(), lr=learning_rate, l1=penalty, max_s=s, soft=soft)
                optimizer_checkpoint = torch.load(os.path.join(model_dir, 'optim.pkl'))
                optimizer_all.load_state_dict(optimizer_checkpoint)

            writer_test = SummaryWriter(log_dir, flush_secs=600, purge_step=net.epochs)
            writer_train = SummaryWriter(log_dir, flush_secs=600, purge_step=net.train_times)
        else:
            writer_test = SummaryWriter(log_dir, flush_secs=600)
            writer_train = SummaryWriter(log_dir, flush_secs=600)
            if no_prune:
                optimizer_all = Adam(net.parameters(), lr=learning_rate)
            else:
                if args.mode == 'grad':
                    optimizer_all = GradRewiring(net.parameters(), lr=learning_rate, alpha=penalty, s=s)
                elif args.mode == 'deep':
                    optimizer_all = DeepRewiring(net.parameters(), lr=learning_rate, l1=penalty, max_s=s, soft=soft)

        print(net)

        max_test_accuracy = 0

        before_link = dict()
        after_link = dict()
        
        # Record initial connectivity
        if not no_prune:
            for name, param in net.named_parameters():
                before_link[name] = (param.abs() >= 1e-10)

        # Training Loop
        for _ in range(N + 1):
            net.train()
            print(f'Epoch {net.epochs}, {file_prefix}')

            time_start = time.time()
            for img, label in train_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                optimizer_all.zero_grad()

                out_spikes_counter = net(img)

                out_spikes_counter_frequency = out_spikes_counter / T

                loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
                loss.backward()

                optimizer_all.step()

                reset_net(net)

                if net.train_times % i2 == 0:
                    correct_rate = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
                    writer_train.add_scalar('train/acc', correct_rate, net.train_times)
                    writer_train.add_scalar('train/loss', loss.item(), net.train_times)

                net.train_times += 1

            # Evaluate at the end of training epoch
            net.eval()

            with torch.no_grad():

                if net.epochs % i1 == 0:
                    set_monitor(net, True)
                    spike_times = dict()
                    for name, module in net.named_modules():
                        if hasattr(module, 'monitor'):
                            spike_times[name] = 0

                test_sum = 0
                correct_sum = 0
                for img, label in test_data_loader:
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)

                    out_spikes_counter = net(img)

                    correct_sum += (out_spikes_counter.argmax(dim=1) == label).float().sum().item()
                    test_sum += label.numel()
                    
                    if net.epochs % i1 == 0:
                        for name, module in net.named_modules():
                            if hasattr(module, 'monitor'):
                                # monitor['s'] is a list, each element is of shape [batch_size, ...]
                                spike_times[name] += torch.sum(torch.from_numpy(np.concatenate(module.monitor['s'], axis=0)).cuda(), dim=0)

                    reset_net(net)

############ 1. Avg. Firing Rate & Proportion of Silent Neurons ###########

                if net.epochs % i1 == 0:
                    for k, v in spike_times.items():
                        
                        nonfire_prop = (v == 0).sum().cpu().item() / v.numel()

                        avg_firing_rate = v.mean() / (test_sum * T)

                        writer_test.add_scalar('nonfire_prop/' + k, nonfire_prop, net.epochs)
                        writer_test.add_scalar('avg_firing_rate/' + k, avg_firing_rate, net.epochs)

###########################################################################

############ 2. Test Acc. ###########

                test_accuracy = correct_sum / test_sum
                writer_test.add_scalar('test_acc', test_accuracy, net.epochs)

#####################################

                if no_prune:
                    print(f'Test Acc: {test_accuracy * 100:.3f}%, Max Test Acc: {max_test_accuracy * 100:.3f}%')
                    if test_accuracy > max_test_accuracy:
                        max_test_accuracy = test_accuracy
                    torch.save(optimizer_all.state_dict(), os.path.join(model_dir, 'optim.pkl'))
                else:
                    zero_cnt = 0.0
                    for name, param in net.named_parameters():
############ 3. Sparsity ############
                        curr_zero_cnt = (param.abs() < 1e-10).float().sum()
                        zero_cnt += curr_zero_cnt
                        writer_test.add_scalar('layer sparsity/' + name, curr_zero_cnt / param.numel(), net.epochs)
#####################################


############ 4. Connection Change ###########
                        after_link[name] = (param.abs() >= 1e-10)

                        regrow_cnt = torch.logical_and(torch.logical_not(before_link[name]), after_link[name]).sum().item()
                        prune_cnt = torch.logical_and(torch.logical_not(after_link[name]), before_link[name]).sum().item()

                        writer_test.add_scalar('regrow_cnt/' + name, regrow_cnt, net.epochs)
                        writer_test.add_scalar('prune_cnt/' + name, prune_cnt, net.epochs)
                        writer_test.add_scalar('prune-regrow/' + name, prune_cnt - regrow_cnt, net.epochs)

                        before_link[name] = after_link[name].clone()
#############################################


############ 5. Distribution of Weight and Hidden Parameters #########
                        if net.epochs % i1 == 0:
                            state = optimizer_all.state[param]
                            writer_test.add_histogram(name + '-w', param, net.epochs)
                            if args.mode != 'deep' or soft:
                                writer_test.add_histogram(name + '-theta', state['strength'], net.epochs)
                                torch.save(state['strength'], os.path.join(model_dir, f'{name}-theta-{net.epochs}.pkl'))
######################################################################

                    sparsity_all = zero_cnt / ttl_cnt

                    writer_test.add_scalar('sparsity', sparsity_all, net.epochs)

                    print(f'Test Acc: {test_accuracy * 100:.3f}%, Sparsity: {sparsity_all * 100:.3f}%')
                    torch.save(optimizer_all.state_dict(), os.path.join(model_dir, 'optim.pkl'))

                torch.save(net, os.path.join(model_dir, 'net.pkl'))

                if net.epochs % i1 == 0:
                    torch.save(net, os.path.join(model_dir, f'net-{net.epochs}.pkl'))
                    set_monitor(net, False)

            net.epochs += 1
            
            time_end = time.time()
            print(f'Elapse: {time_end - time_start:.2f}s')

            if net.epochs > N:
                break
