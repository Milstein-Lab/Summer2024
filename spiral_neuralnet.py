# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import pickle
import sys
import os
import click
from os import cpu_count
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from functools import partial
import traceback
import time
from datetime import datetime


# set_seed() and seed_worker()
def set_seed(seed=None, seed_torch=True, verbose=False):
    """
    Function that controls randomness. NumPy and random modules must be imported.
  
    Args:
    - seed (Integer): A non-negative integer that defines the random state. Default is `None`.
    - seed_torch (Boolean): If `True` sets the random seed for pytorch tensors, so pytorch module
                            must be imported. Default is `True`.
    - verbose (boolean): If True, print seed being used.
  
    Returns:
      Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if verbose:
        print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
    """
    DataLoader will reseed workers following randomness in
    multi-process data loading algorithm.
  
    Args:
    - worker_id (integer): ID of subprocess to seed. 0 means that the data will be loaded in the main process
                            Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details
  
    Returns:
      Nothing
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed()
    
# set_device() to CPU or GPU
def set_device(verbose=False):
    """
    Set the device. CUDA if available, CPU otherwise
  
    Args:
    - verbose (boolean): If True, print whether GPU is being used.
  
    Returns:
    - Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        if device != "cuda":
            print("GPU is not enabled in this notebook")
        else:
            print("GPU is enabled in this notebook")

    return device

# Create spiral dataset
def create_spiral_dataset(K, sigma, N):
    """
    Function to simulate spiral dataset
  
    Args:
    - K (int): Number of classes
    - sigma (float): Standard deviation
    - N (int): Number of data points
  
    Returns:
    - X (torch.tensor): Spiral data
    - y (torch.tensor): Corresponding ground truth
    """

    # Initialize t, X, y
    t = torch.linspace(0, 1, N)
    X = torch.zeros(K*N, 2)
    y = torch.zeros(K*N)

    # Create data
    for k in range(K):
        X[k*N:(k+1)*N, 0] = t*(torch.sin(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))
        X[k*N:(k+1)*N, 1] = t*(torch.cos(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))
        y[k*N:(k+1)*N] = k

    return X, y


# Net class
class Net(nn.Module):
    """
    Simulate MLP Network
    """

    def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num, lr, extra_params=None,
                 description=None, use_bias=True, learn_bias=True, mean_subtract_input=False, seed=0):
        """
        Initialize MLP Network parameters
    
        Args:
        - actv (string): Activation function
        - input_feature_num (int): Number of input features
        - hidden_unit_nums (list): Number of units per hidden layer. List of integers
        - output_feature_num (int): Number of output features
        - lr (float): Learning rate
        - description (string): Learning rule to use
        - use_bias (boolean): If True, randomly initialize biases. If False, set all biases to 0.
        - learn_bias (boolean): If True, use learning rule to update biases. If False, otherwise.
        - mean_subtract_input (boolean): If True, forward method mean subtracts input before computing output
        Returns:
        - Nothing
        """
        super(Net, self).__init__()

        torch.manual_seed(seed)

        self.input_feature_num = input_feature_num
        self.hidden_unit_nums = hidden_unit_nums
        self.output_feature_num = output_feature_num
        self.description = description
        self.use_bias = use_bias
        self.learn_bias = learn_bias
        self.mean_subtract_input = mean_subtract_input
        self.lr = lr
        self.extra_params = extra_params

        self.forward_soma_state = {} # states of all layers pre activation
        self.forward_activity = {} # activities of all layers post ReLU
        if self.mean_subtract_input:
            self.forward_activity_mean_subtracted = {}
        self.weights = {}
        self.biases = {}
        self.initial_weights = {}
        self.initial_biases = {}
        self.activation_functions = {}
        self.layers = {}

        self.forward_dend_state = {}
        self.backward_dend_state = {}
        self.nudges = {}
        self.backward_activity = {}
        
        self.recurrent_layers = {}
        self.recurrent_weights = {}
        if 'dend_EI_contrast' in self.description:
            for i, num in enumerate(self.hidden_unit_nums):
                layer = f'H{i+1}'
                self.recurrent_layers[layer] = nn.Linear(num, num, bias=False)
                self.recurrent_weights[layer] = self.recurrent_layers[layer].weight
        elif 'oja' in self.description:
            self.running_average_len = 600

        self.hooked_grads = {}

        layers = []
        prev_size = input_feature_num
        for i, hidden_size in enumerate(hidden_unit_nums):
            layer = nn.Linear(prev_size, hidden_size, bias=use_bias)
            layers.append(layer)
            key = f'H{i+1}'
            self.layers[key] = layer

            self.weights[key] = layer.weight
            if use_bias:
                self.biases[key] = layer.bias
                if not learn_bias:
                    layer.bias.requires_grad = False
            else:
                self.biases[key] = torch.zeros(hidden_size)
            self.initial_weights[key] = layer.weight.data.clone()
            self.initial_biases[key] = self.biases[key].data.clone()

            act_layer = actv()
            layers.append(act_layer)
            self.activation_functions[key] = act_layer
            
            prev_size = hidden_size

        out_layer = nn.Linear(prev_size, output_feature_num, bias=use_bias)
        self.layers['Out'] = out_layer
        layers.append(out_layer) # Output state layer
        self.weights['Out'] = out_layer.weight
        if use_bias:
            self.biases['Out'] = out_layer.bias
            if not learn_bias:
                out_layer.bias.requires_grad = False
        else:
            self.biases['Out'] = torch.zeros(output_feature_num)
        self.initial_weights['Out'] = out_layer.weight.data.clone()
        self.initial_biases['Out'] = self.biases['Out'].data.clone()

        last_layer = actv()
        layers.append(last_layer) # ReLU after output state layer
        self.activation_functions['Out'] = last_layer

        self.mlp = nn.Sequential(*layers)

        if 'backprop' in self.description:
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x, store=True, testing=True,):
        """
        Simulate forward pass of MLP Network
    
        Args:
        - x (torch.tensor): Input data
        - store (boolean): If True, store intermediate states and activities of each layer
        - test (boolean): If True, expect full batch to be contained in x
    
        Returns:
        - x (torch.tensor): Output data
        """
        if store:
            self.forward_activity['Input'] = x.detach().clone()
        
        if self.mean_subtract_input:
            if not testing:
                if self.forward_activity_train_history['Input']:
                    x = x - torch.mean(torch.stack(self.forward_activity_train_history['Input'][-self.running_average_len:]), dim=0)
                    self.forward_activity_mean_subtracted['Input'] = x.detach().clone()
                else:
                    self.forward_activity_mean_subtracted['Input'] = x.detach().clone()
            else:
                x = x - torch.mean(x, dim=0)
                self.forward_activity_mean_subtracted['Input'] = x.detach().clone()
        
        for key, layer in self.layers.items():
            x = layer(x)
            if store:
                self.forward_soma_state[key] = x.detach().clone() # Before ReLU
            
            x = self.activation_functions[key](x)

            if store:
                self.forward_activity[key] = x.detach().clone() # After ReLU
            
            if self.mean_subtract_input:
                if not testing:
                    if self.forward_activity_train_history[key]:
                        x = x - torch.mean(torch.stack(self.forward_activity_train_history[key][-self.running_average_len:]), dim=0)
                        self.forward_activity_mean_subtracted[key] = x.detach().clone()
                    else:
                        self.forward_activity_mean_subtracted[key] = x.detach().clone()
                else:
                    x = x - torch.mean(x, dim=0)
                    self.forward_activity_mean_subtracted[key] = x.detach().clone()


        return x

    def save_gradients(self, key):
        def hook_fn(module, grad_input, grad_output):
            self.hooked_grads[key] = grad_output[0]
        return hook_fn

    def register_hooks(self):
        for key in self.layers.keys():
            self.layers[key].register_full_backward_hook(self.save_gradients(key))
    
    def test(self, data_loader, device='cpu', store=True, testing=True):
        """
        Function to gauge network performance
    
        Args:
        - data_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset
        - device (string): CUDA/GPU if available, CPU otherwise
    
        Returns:
        - total (int): Number of datapoints in the dataloader
        - acc (float): Performance of the network
        """
        correct = 0
        total = 0
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = self.forward(inputs, store=store, testing=True)
            _, predicted = torch.max(outputs, 1)
            if store:
                self.test_labels = labels
                self.predicted_labels = predicted
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        return total, acc
    
    def reinit(self):
        self.training_losses = []
        
        self.train_accuracy = []
        self.train_steps_list = []
        self.avg_loss = []
        
        self.forward_soma_state_train_history = {}
        self.forward_activity_train_history = {}
        self.backward_activity_train_history = {}
        if self.mean_subtract_input:
            self.forward_activity_mean_subtracted_train_history = {}
        self.forward_dend_state_train_history = {}
        self.backward_dend_state_train_history = {}
        self.nudges_train_history = {}
        self.weights_train_history = {}
        self.biases_train_history = {}
        self.recurrent_weights_train_history = {}
        self.forward_activity_train_history['Input'] = []
        if self.mean_subtract_input:
            self.forward_activity_mean_subtracted_train_history['Input'] = []
        for key in self.layers:
            self.forward_soma_state_train_history[key] = []
            self.forward_activity_train_history[key] = []
            if self.mean_subtract_input:
                self.forward_activity_mean_subtracted_train_history[key] = []
            self.backward_activity_train_history[key] = []
            self.forward_dend_state_train_history[key] = []
            self.backward_dend_state_train_history[key] = []
            self.nudges_train_history[key] = []
            self.weights_train_history[key] = []
            self.recurrent_weights_train_history[key] = []
            if self.use_bias or self.learn_bias:
                self.biases_train_history[key] = []
        
        self.train_labels = []
        self.predicted_labels = []
    
    def train_model(self, description, train_loader, val_loader, debug=False, num_train_steps=None, num_epochs=1, verbose=False, device='cpu'):
        """
        Train model with backprop, accumulate loss, evaluate performance
    
        Args:
        - description (string): Description of model to train
        - train_loader (torch.utils.data type): Combines the train dataset and sampler, and provides an iterable over the given dataset
        - val_loader (torch.utils.data type): Contains a validation dataset as a single batch
        - debug (boolean): If True, enters debug mode.
        - num_train_steps (int): Stops train loop after specified number of steps
        - num_epochs (int): Number of epochs [default: 1]
        - verbose (boolean): If True, print statistics
        - device (string): CUDA/GPU if available, CPU otherwise
    
        Returns:
        - val_acc (int): Accuracy of model on train data
        """
        self.to(device)
        self.train()
        self.reinit()
        train_step = 0
        criterion_function = eval(f"nn.MSELoss()")

        for epoch in range(num_epochs):  # Loop over the dataset multiple times
            for data in tqdm(train_loader):
                # Get the inputs; data is a list of [input, label]
                input, label = data
                input = input.to(device).float()
                label = label.to(device).long()

                self.train_labels.append(label.item())

                # forward pass
                outputs = self.forward(input, testing=False)
                _, predicted = torch.max(outputs, 1)
                targets = torch.eye(self.output_feature_num)[label]

                self.predicted_labels.append(predicted)

                loss = criterion_function(outputs, targets)

                # Track losses
                self.training_losses.append(loss.item())

                if train_step % 100 == 0:
                    correct = (torch.tensor(self.predicted_labels[-100:]) == torch.tensor(self.train_labels[-100:])).sum().item()
                    acc = correct
                    self.train_accuracy.append(acc)
                    self.train_steps_list.append(train_step) 
                    self.avg_loss.append(torch.mean(torch.tensor(self.training_losses[-100:])))                 
                
                # Choose learning rule
                if 'backprop' in description:
                    self.train_backprop(loss)
                elif 'dend' in description:
                    self.train_dend(description, targets)
                
                self.store_train_history()
                
                # Stop after one certain number of train step
                train_step += 1
                if num_train_steps is not None:
                    if train_step == num_train_steps:
                        assert False

        self.stack_train_history()
        
        self.eval()

        val_total, val_acc = self.test(val_loader, device, testing=True)
        self.val_acc = val_acc
        self.final_loss = self.training_losses[-1]

        # Store final weights and biases
        self.final_weights = {}
        self.final_biases = {}
        for key, layer in self.layers.items():
            self.final_weights[key] = layer.weight.data.clone()
            self.final_biases[key] = self.biases[key].data.clone()

        if verbose:
            print(f'\nAccuracy on the {val_total} training samples: {val_acc:0.2f}')

        return val_acc
    
    def train_backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ReLU_derivative(self, x):
        output = torch.ones_like(x)
        indexes = torch.where(x <= 0)
        output[indexes] = 0
        return output

    def backward_dend_temp_contrast(self, targets):
        with torch.no_grad():
            prev_layer = None
            reverse_layers = list(self.layers.keys())[::-1]
            
            for layer in reverse_layers:
                if layer == 'Out':
                    self.nudges[layer] = (2.0 / self.output_feature_num) * self.ReLU_derivative(
                        self.forward_soma_state[layer]) * (targets - self.forward_activity['Out'])  # the ReLU derivative term is dA/dz
                    self.nudges_train_history[layer].append(self.nudges[layer])
                else:
                    self.forward_dend_state[layer] = self.forward_activity[prev_layer] @ self.weights[prev_layer]
                    self.backward_dend_state[layer] = self.backward_activity[prev_layer] @ self.weights[prev_layer]
                    self.nudges[layer] = self.ReLU_derivative(self.forward_soma_state[layer]) * (
                            self.backward_dend_state[layer] - self.forward_dend_state[layer])
                    self.forward_dend_state_train_history[layer].append(self.nudges[layer])
                    self.backward_dend_state_train_history[layer].append(self.backward_dend_state[layer])
                    self.nudges_train_history[layer].append(self.nudges[layer])
                
                self.backward_activity[layer] = self.activation_functions[layer](self.forward_soma_state[layer] + self.nudges[layer])
                prev_layer = layer

    def step_dend_temp_contrast(self):
        # add beta scalars?
        with torch.no_grad():
            lr = self.lr
            with torch.no_grad():
                prev_layer = 'Input'
                for layer in self.layers.keys():
                    self.weights[layer].data += lr * torch.outer(self.nudges[layer].squeeze(),
                                                                self.forward_activity[prev_layer].squeeze())
                    if self.use_bias and self.learn_bias:
                        self.biases[layer].data += lr * self.nudges[layer].squeeze()
                    prev_layer = layer

    def backward_ojas(self, targets):
        with torch.no_grad():
            beta_Out = self.extra_params['beta_Out']
            beta_H2 = self.extra_params['beta_H2']
            beta_H1 = self.extra_params['beta_H1']
            #beta_Input = self.extra_params['beta_Input']



            prev_layer = None
            reverse_layers = list(self.layers.keys())[::-1]
            
            for layer in reverse_layers:
                if layer == 'Out':
                    self.nudges[layer] = beta_Out * self.ReLU_derivative(self.forward_soma_state[layer]) * (targets - self.forward_activity['Out'])  # the ReLU derivative term is dA/dz
                    self.nudges_train_history[layer].append(self.nudges[layer])
                else:
                    self.forward_dend_state[layer] = self.forward_activity[prev_layer] @ self.weights[prev_layer]
                    self.backward_dend_state[layer] = self.backward_activity[prev_layer] @ self.weights[prev_layer]
                    self.nudges[layer] = eval(f'beta_{layer}') * self.ReLU_derivative(self.forward_soma_state[layer]) * (self.backward_dend_state[layer] - self.forward_dend_state[layer])
                    self.forward_dend_state_train_history[layer].append(self.nudges[layer])
                    self.backward_dend_state_train_history[layer].append(self.backward_dend_state[layer])
                    self.nudges_train_history[layer].append(self.nudges[layer])
                
                self.backward_activity[layer] = self.activation_functions[layer](
                    self.forward_soma_state[layer] + self.nudges[layer])
                prev_layer = layer
    
    def step_ojas(self):
        with torch.no_grad():
            if 'lr_Out' in self.extra_params:
                lr_Out = self.extra_params['lr_Out']
            else:
                lr_Out = self.lr
            if 'lr_H2' in self.extra_params:
                lr_H2 = self.extra_params['lr_H2']
            else:
                lr_H2 = self.lr
            if 'lr_H1' in self.extra_params:
                lr_H1 = self.extra_params['lr_H1']
            else:
                lr_H1 = self.lr
            # lr_Input = self.lr
            alpha_Out = self.extra_params['alpha_Out']
            alpha_H2 = self.extra_params['alpha_H2']
            alpha_H1 = self.extra_params['alpha_H1']
            #alpha_Input = self.extra_params['alpha_Input']

            with torch.no_grad():
                prev_layer = 'Input'
                for layer in self.layers.keys():
                    self.weights[layer].data += (self.lr * self.backward_activity[layer].T * (self.forward_activity_mean_subtracted[prev_layer] - eval(f'alpha_{layer}') * self.backward_activity[layer].T * self.weights[layer].data))
                    if self.use_bias and self.learn_bias:
                        self.biases[layer].data += self.lr * self.nudges[layer].squeeze()
                    prev_layer = layer
    
    def backward_dend_EI_contrast(self, targets):
        with torch.no_grad():
            reverse_layers = list(self.layers.keys())[::-1]
            for idx, layer in enumerate(reverse_layers):
                if layer == 'Out':
                    self.nudges[layer] = (2.0 / self.output_feature_num) * self.ReLU_derivative(self.forward_soma_state[layer]) * (targets - self.forward_activity['Out'])
                    self.nudges[layer].clamp_(-1., 1.)
                else:
                    upper_layer = reverse_layers[idx - 1]
                    self.forward_dend_state[layer] = self.forward_activity[upper_layer] @ self.weights[upper_layer] + self.recurrent_layers[layer](self.forward_activity[layer])
                    self.forward_dend_state[layer].clamp_(-1., 1.)
                    self.backward_dend_state[layer] = self.backward_activity[upper_layer] @ self.weights[upper_layer] + self.recurrent_layers[layer](self.forward_activity[layer])
                    self.backward_dend_state[layer].clamp_(-1., 1.)
                    self.nudges[layer] = self.backward_dend_state[layer] * self.ReLU_derivative(self.forward_soma_state[layer])
                self.backward_activity[layer] = self.activation_functions[layer](self.forward_soma_state[layer] + self.nudges[layer])

    def step_dend_EI_contrast(self):
        with torch.no_grad():
            lr = self.lr
            lower_layer = 'Input'
            for idx, layer in enumerate(self.layers.keys()):
                if layer != 'Out':
                    rec_lr_key = f'rec_lr_H{idx+1}'
                    rec_lr = self.extra_params[rec_lr_key]

                self.weights[layer].data += lr * torch.outer(self.nudges[layer].squeeze(), torch.clamp(self.forward_activity[lower_layer].squeeze(), 0., 1.))
                
                if layer != 'Out':
                    self.recurrent_weights[layer].data += -1 * rec_lr * self.forward_dend_state[layer].T @ torch.clamp(self.forward_activity[layer], 0., 1.)
                
                if self.use_bias and self.learn_bias:
                    self.biases[layer].data += lr * self.nudges[layer].squeeze()
                
                lower_layer = layer

    def store_train_history(self):
        # Store forward state and activity info
        self.forward_activity_train_history['Input'].append(self.forward_activity['Input'])
        for key in self.layers:
            self.forward_soma_state_train_history[key].append(self.forward_soma_state[key])
            self.forward_activity_train_history[key].append(self.forward_activity[key])
            if self.mean_subtract_input:
                self.forward_activity_mean_subtracted_train_history[key].append(
                    self.forward_activity_mean_subtracted[key])
                
            # store a copy of the weights and biases
            self.weights_train_history[key].append(self.weights[key].data.clone())
            if self.use_bias or self.learn_bias:
                self.biases_train_history[key].append(self.biases[key].data.clone())
            if key in self.recurrent_weights:
                self.recurrent_weights_train_history[key].append(self.recurrent_weights[key].data.clone())
            
            if key in self.nudges:
                self.nudges_train_history[key].append(self.nudges[key])
            if key in self.forward_dend_state:
                self.forward_dend_state_train_history[key].append(self.forward_dend_state[key])
            if key in self.backward_dend_state:
                self.backward_dend_state_train_history[key].append(self.backward_dend_state[key])
            if key in self.backward_activity:
                self.backward_activity_train_history[key].append(self.backward_activity[key])
    
    def stack_train_history(self):
        # Stack all history tensors
        for key in self.layers:
            self.forward_soma_state_train_history[key] = torch.stack(self.forward_soma_state_train_history[key]).squeeze()
            self.forward_activity_train_history[key] = torch.stack(self.forward_activity_train_history[key]).squeeze()
            if self.mean_subtract_input:
                self.forward_activity_mean_subtracted_train_history[key] = torch.stack(self.forward_activity_mean_subtracted_train_history[key]).squeeze()
            if self.forward_dend_state_train_history[key]:
                self.forward_dend_state_train_history[key] = torch.stack(self.forward_dend_state_train_history[key]).squeeze()
            if self.backward_dend_state_train_history[key]:
                self.backward_dend_state_train_history[key] = torch.stack(self.backward_dend_state_train_history[key]).squeeze()
            if self.nudges_train_history[key]:
                self.nudges_train_history[key] = torch.stack(self.nudges_train_history[key]).squeeze()
            if self.backward_activity_train_history[key]:
                self.backward_activity_train_history[key] = (torch.stack(self.backward_activity_train_history[key]).squeeze())
            if self.weights_train_history[key]:
                self.weights_train_history[key] = torch.stack(self.weights_train_history[key]).squeeze()
            if self.recurrent_weights_train_history[key]:
                self.recurrent_weights_train_history[key] = (torch.stack(self.recurrent_weights_train_history[key]).squeeze())
            if self.use_bias and self.learn_bias:
                if self.biases_train_history[key]:
                    self.biases_train_history[key] = torch.stack(self.biases_train_history[key]).squeeze()
        
        self.train_labels = torch.tensor(self.train_labels)
        self.training_losses = torch.tensor(self.training_losses)
        self.train_steps_list = torch.tensor(self.train_steps_list)
        self.train_accuracy = torch.tensor(self.train_accuracy)
    
    def train_dend(self, description, targets):
        '''
        Wrapper function for training models with dendritic learning rules
        - Dendritic Temporal Contrast
        - Oja's Rule
        - Dendritic Excitatory-Inhibitory (EI) Contrast

        Args:
        - description (string): Description of model to train
        - targets (torch tensor): Target activities for output neurons
        - lr (float): Learning rate

        Returns:
        - Nothing
        '''
        self.eval()
        if 'dend_temp_contrast' in description:
            self.backward_dend_temp_contrast(targets)
            self.step_dend_temp_contrast()
        elif 'oja' in description:
            self.backward_ojas(targets)
            self.step_ojas()
        elif 'dend_EI_contrast' in description:
            self.backward_dend_EI_contrast(targets)
            self.step_dend_EI_contrast()
    
    def test_model(self, test_loader, verbose=True, device='cpu'):
        '''
        Evaluate performance

        Args:
        - test_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset
        - verbose (boolean): If True, print statistics
        - device (string): CUDA/GPU if available, CPU otherwise

        Returns:
        - test_acc (int): Accuracy of model on test data
        '''
        self.to(device)
        test_total, test_acc = self.test(test_loader, device)

        if verbose:
            print(f'Accuracy on the {test_total} testing samples: {test_acc:0.2f}\n')
        
        self.test_acc = test_acc
        return test_acc
    
    def get_decision_map(self, DEVICE='cpu', M=500, map_extent=2.):
        """
        Helper function to plot decision map
    
        Args:
        - X_test (torch.tensor): Test data
        - y_test (torch.tensor): Labels of the test data
        - DEVICE (cpu or gpu): Device type
        - M (int): Size of the constructed tensor with meshgrid
        - eps (float): Decision threshold
    
        Returns:
        - decision_map.T (torch.Tensor): Decision map transpose to use in graph
        """
        X_all = sample_grid(M, x_max=map_extent)
        print('sample_grid_shape:', X_all.shape)
        y_pred = self.forward(X_all.to(DEVICE), store=False).cpu()

        decision_map = torch.argmax(y_pred, dim=1)
        decision_map = decision_map.view(M, M)
        return decision_map.T

    def display_summary(self, model_dict, title=None, seed=None, png_save_path=None, svg_save_path=None, show_plot=False):
        '''
        Display network summary

        Args:
        - title (string): Title of model based on description
        - seed (int): network_seed used to create model
        - png_save_path (string): File path to save plot as png
        - svg_save_path (string): File path to save plot as svg
        - show_plot (boolean): If True, shows the plot

        Returns:
        - Nothing
        '''

        inputs = self.forward_activity['Input']
        test_labels = self.test_labels
        predicted_labels = self.predicted_labels

        class_averaged_activity = {}
        sorted_indices_layers = {}
        for key, activity in self.forward_activity.items():
            this_class_averaged_activity = torch.empty((self.output_feature_num, self.forward_activity[key].shape[1]))
            for label in torch.arange(self.output_feature_num):
                indexes = torch.where(test_labels == label)
                this_class_averaged_activity[label,:] = torch.mean(activity[indexes], dim=0)

            max_indices = this_class_averaged_activity.argmax(dim=0)
            if key == 'Out':
                this_sorted_indices = torch.arange(self.output_feature_num)
            else:
                values_sorted, this_sorted_indices = torch.sort(max_indices, stable=True)

            class_averaged_activity[key] = this_class_averaged_activity
            sorted_indices_layers[key] = this_sorted_indices

        num_layers = max(2, len(self.hidden_unit_nums)+1)
        fig, axes = plt.subplots(nrows=2, ncols=num_layers, figsize=(3*num_layers, 6))

        for i, key in enumerate(self.layers):
            this_class_averaged_activity = class_averaged_activity[key]
            this_sorted_indices = sorted_indices_layers[key]
            imH = axes[0][i].imshow(this_class_averaged_activity[:,this_sorted_indices].T, aspect='auto', interpolation='none')
            if key == 'Out':
                axes[0][i].set_title(f'Output Layer')
                axes[0][i].set_yticks(range(this_class_averaged_activity.shape[1]))
                axes[0][i].set_yticklabels(range(this_class_averaged_activity.shape[1]))
            else:
                axes[0][i].set_title(f'Hidden Layer {i+1}')
            axes[0][i].set_xlabel('Label')
            axes[0][i].set_ylabel('Neuron')
            fig.colorbar(imH, ax=axes[0][i])

            axes[0][i].set_xticks(range(this_class_averaged_activity.shape[0]))
            axes[0][i].set_xticklabels(range(this_class_averaged_activity.shape[0]))


        #average accuracies and losses with STD shaded
        results = model_dict[self.description]

        # Accuracies
        all_train_accuracies = [results[seed].train_accuracy for seed in results]
        all_val_accuracies = [results[seed].val_acc for seed in results]
        all_test_accuracies = [results[seed].test_acc for seed in results]

        train_steps_list = results[next(iter(results))].train_steps_list  # Assuming all have the same train steps

        # Average accuracies and standard deviations
        average_train_accuracies = np.mean(all_train_accuracies, axis=0)
        std_train_accuracies = np.std(all_train_accuracies, axis=0)

        average_val_acc = np.mean(all_val_accuracies)
        std_val_acc = np.std(all_val_accuracies)

        average_test_acc = np.mean(all_test_accuracies)
        std_test_acc = np.std(all_test_accuracies)



        # Losses
        all_train_losses = [results[seed].avg_loss for seed in results]
        train_steps_list = results[next(iter(results))].train_steps_list  # Assuming all have the same train steps

        # Average losses and standard deviations
        average_train_losses = np.mean(all_train_losses, axis=0)
        final_avg_loss = average_train_losses[-1]
        std_train_losses = np.std(all_train_losses, axis=0)

        # Plotting Accuracies
        axes[1][0].plot(train_steps_list, average_train_accuracies, '-', label=f"Avg Test Acc: {average_test_acc:.3f}\nAvg Val Acc: {average_val_acc:.3f}")
        axes[1][0].fill_between(train_steps_list, average_train_accuracies - std_train_accuracies,
                         average_train_accuracies + std_train_accuracies, alpha=0.2)

        axes[1][0].set_xlabel('Train Steps')
        axes[1][0].set_ylabel('Accuracy (%)')
        axes[1][0].legend(loc='best', frameon=False, handlelength=0)

        # Plotting Losses

        axes[1][1].plot(train_steps_list, average_train_losses, '-', label=f'Avg Final Loss: {final_avg_loss:.3f}')
        axes[1][1].fill_between(train_steps_list, average_train_losses - std_train_losses,
                         average_train_losses + std_train_losses,
                         alpha=0.2)
        axes[1][1].set_xlabel('Train Steps')
        axes[1][1].set_ylabel('Loss')
        axes[1][1].legend(loc='best', frameon=False, handlelength=0)


        # map = self.get_decision_map()
        # axes[1][2].imshow(map, extent=[-2, 2, -2, 2], origin='lower', cmap='coolwarm', alpha=0.7)


        correct_indices = (predicted_labels == test_labels).nonzero().squeeze()
        axes[1][2].scatter(inputs[correct_indices,0], inputs[correct_indices,1], c=test_labels[correct_indices], s=4)
        wrong_indices = (predicted_labels != test_labels).nonzero().squeeze()
        axes[1][2].scatter(inputs[wrong_indices, 0], inputs[wrong_indices, 1], c='darkgrey', s=4)
        axes[1][2].set_xlabel('x1')
        axes[1][2].set_ylabel('x2')
        axes[1][2].set_title('Predictions')

        for j in range(3, num_layers):
            axes[1][j].axis('off')

        if title is not None:
            fig.suptitle(f'Class Averaged Activity - {title}')
        else:
            fig.suptitle('Class Averaged Activity')

        if seed is not None:
            fig.text(0.95, 0.95, f'Seed: {seed}', ha='center', fontsize=12)

        fig.tight_layout(rect=[0, 0, 1, 0.95]) 

        if png_save_path is not None:
            fig.savefig(f'{png_save_path}/summary_{self.description}.png', bbox_inches='tight', format='png')
        if svg_save_path is not None:
            fig.savefig(f'{svg_save_path}/summary_{self.description}.svg', bbox_inches='tight', format='svg')
        if show_plot:
            fig.show()

    def plot_params(self, title=None, seed=None, png_save_path=None, svg_save_path=None, show_plot=False):
        '''
        Plot initial and final weights and biases for all layers

        Args:
        - title (string): Title of model based on description
        - seed (int): network_seed used to create model
        - png_save_path (string): File path to save plot as png
        - svg_save_path (string): File path to save plot as svg
        - show_plot (boolean): If True, shows the plot

        Returns:
        - None
        '''

        num_layers = max(2, len(self.layers))
        fig, axes = plt.subplots(nrows=2, ncols=num_layers, figsize=(3*num_layers, 6))

        for i, key in enumerate(self.layers):
            iw = self.initial_weights[key].flatten()
            fw = self.final_weights[key].flatten()

            axes[0][i].hist([iw, fw], bins=30, label=['Initial Weights', 'Final Weights'])

            if key == 'Out':
                axes[0][i].set_title('Output Layer')
            else:
                axes[0][i].set_title(f'Hidden Layer {i+1}')

            if i == 0:
                axes[0][0].legend()
                axes[0][0].set_ylabel(f'Weights\nFrequency')
            axes[0][i].set_xlabel("Weight Value")


            ib = self.initial_biases[key].flatten()
            fb = self.final_biases[key].flatten()

            axes[1][i].hist([ib, fb], bins=30, label=['Initial Biases', 'Final Biases'])

            if key == 'Out':
                axes[1][i].set_title('Output Layer')
            else:
                axes[1][i].set_title(f'Hidden Layer {i+1}')

            if i == 0:
                axes[1][0].legend()
                axes[1][0].set_ylabel(f'Biases\nFrequency')
            axes[1][i].set_xlabel("Bias Value")
            
        if title is not None:
            fig.suptitle(title)

        if seed is not None:
            fig.text(0.95, 0.95, f'Seed: {seed}', ha='center', fontsize=12)

        fig.tight_layout(rect=[0, 0, 1, 0.95])  

        if png_save_path is not None:
            fig.savefig(f'{png_save_path}/params_{self.description}.png', bbox_inches='tight', format='png')
        if svg_save_path is not None:
            fig.savefig(f'{svg_save_path}/params_{self.description}.svg', bbox_inches='tight', format='svg')
        if show_plot:
            fig.show()


def sample_grid(M, x_max=2.0):
    """
    Helper function to simulate sample meshgrid
  
    Args:
    - M (int): Size of the constructed tensor with meshgrid
    - x_max (float): Defines range for the set of points
  
    Returns:
    - X_all (torch.tensor): Concatenated meshgrid tensor
    """
    ii, jj = torch.meshgrid(torch.linspace(-x_max, x_max, M),
                            torch.linspace(-x_max, x_max, M),
                            indexing="ij")
    X_all = torch.cat([ii.unsqueeze(-1),
                       jj.unsqueeze(-1)],
                      dim=-1).view(-1, 2)
    return X_all

def generate_data(K=4, sigma=0.16, N=2000, seed=None, gen=None, display=False, png_save_path=None, svg_save_path=None):
    '''
    Generate spiral dataset for training, testing, and validating a neural network

    Args:
    - K (int): Number of classes in the dataset. Default is 4
    - sigma (float): Standard deviation of the spiral dataset. Default is 0.16
    - N (int): Number of samples in the dataset. Default is 2000
    - seed (int): Seed value for reproducibility. Default is None
    - gen (torch.Generator): Generator object for random number generation. Default is None.
    - display (bool): Whether to display a scatter plot of the dataset. Default is False.
    - png_save_path (string): File path to save plot as png
    - svg_save_path (string): File path to save plot as svg

    Returns:
    - X_test (torch.Tensor): Test input data
    - y_test (torch.Tensor): Test target data
    - X_train (torch.Tensor): Train input data
    - y_train (torch.Tensor): Train target data
    - X_val (torch.Tensor): Validation input data
    - y_val (torch.tensor): Validation target data
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data
    - train_loader (torch.utils.data.DataLoader): DataLoader for train data
    - val_loader (torch.utils.data.DataLoader): Dataloader for validation data
    '''

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Spiral Data Set graph
    X, y = create_spiral_dataset(K, sigma, N)
    
    num_samples = X.shape[0]
    # Shuffle data
    shuffled_indices = torch.randperm(num_samples)   # Get indices to shuffle data
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Split data into train/test
    test_size = int(0.15 * num_samples)
    val_size = int(0.15 * num_samples)
    train_size = num_samples - (test_size + val_size)

    test_end_idx = test_size
    val_end_idx = test_size + val_size

    X_test = X[:test_end_idx]
    y_test = y[:test_end_idx]
    X_val = X[test_end_idx:val_end_idx]
    y_val = y[test_end_idx:val_end_idx]
    X_train = X[val_end_idx:]
    y_train = y[val_end_idx:]

    fig = None
    if display or png_save_path is not None or svg_save_path is not None:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c = y_train, s=10)
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        axes[0].set_title('Train Data')

        axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10)
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].set_title('Test Data')

        axes[2].scatter(X_val[:, 0], X_val[:, 1], c=y_val, s=10)
        axes[2].set_xlabel('x1')
        axes[2].set_ylabel('x2')
        axes[2].set_title('Validation Data')

        fig.tight_layout()

    if display:
        fig.show()
    if png_save_path is not None:
        fig.savefig(f'{png_save_path}/data.png', bbox_inches='tight', format='png')
    if svg_save_path is not None:
        fig.savefig(f'{svg_save_path}/data.svg', bbox_inches='tight', format='svg')

    # Train and test DataLoaders
    if gen is None:
        gen = torch.Generator()
    batch_size = 1
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0)

    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=0)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0, generator=gen)
    
    return X_test, y_test, X_train, y_train, X_val, y_val, test_loader, train_loader, val_loader

def evaluate_model(base_seed, num_input_units, hidden_units, num_classes, description, lr, debug, num_train_steps, show_plot=False, 
                   png_save_path=None, svg_save_path=None, test=False, plot_example_seed=None, extra_params=None, return_net=False):
    
    num_epochs = 1
    data_split_seed = 0
    network_seed = base_seed + 1
    data_order_seed = base_seed + 2
    DEVICE = set_device()
    local_torch_random = torch.Generator()
    local_torch_random.manual_seed(data_order_seed)

    if base_seed != plot_example_seed:
        show_plot = False
        png_save_path = None
        svg_save_path = None
        
    if debug: print(os.getpid()) 
    _, _, _, _, _, _, test_loader, train_loader, val_loader = (
        generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=show_plot,
                      png_save_path=png_save_path, svg_save_path=svg_save_path))
    
    if "ojas_dend" in description:
        mean_subtract_input = True
    else:
        mean_subtract_input = False
    if "learned_bias" in description:
        use_bias = True
        learn_bias = True
    elif "zero_bias" in description:
        use_bias = False
        learn_bias = False
    elif "fixed_bias" in description:
        use_bias = True
        learn_bias = False
    
    net = Net(nn.ReLU, num_input_units, hidden_units, num_classes, description=description,
              use_bias=use_bias, learn_bias=learn_bias, lr=lr, extra_params=extra_params,
              mean_subtract_input=mean_subtract_input, seed=network_seed).to(DEVICE)

    if debug:
        net.register_hooks()
        try:
            net.train_model(description, train_loader, val_loader, debug=debug, num_train_steps=num_train_steps,
                        num_epochs=num_epochs, device=DEVICE)
        except AssertionError:
            print(f"{num_train_steps} train steps completed.")
        except Exception as e:
            traceback.print_exc()
    else:
        net.train_model(description, train_loader, val_loader, debug=debug, num_train_steps=num_train_steps,
                        num_epochs=num_epochs, device=DEVICE)
        
    if test:    
        net.test_model(test_loader, verbose=False, device=DEVICE)
        test_acc = net.test_acc
    else:
        test_acc = None

    val_acc = net.val_acc
    final_val_loss = net.final_loss
    if debug:
        print(os.getpid(), base_seed, 'val_acc', val_acc, 'final_val_loss', final_val_loss,
              'final_train_acc:', net.train_accuracy[-1])
        if test:
            print('test_acc', test_acc)
    if return_net:
        return net, val_acc, final_val_loss, test_acc
    else:
        return None, val_acc, final_val_loss, test_acc


def eval_model_multiple_seeds(description, lr, base_seed, num_seeds, num_cores, num_input_units, hidden_units, num_classes, export,
                              export_file_path, show_plot, png_save_path, svg_save_path, label_dict, debug,
                              num_train_steps, test=True, extra_params=None, verbose=True, return_net=False, **kwargs):
    
    # Determine number of available cores
    if num_cores is None:
        num_cores = min(cpu_count(), num_seeds)
    else:
        num_cores = min(num_cores, num_seeds)
    
    if show_plot and num_cores > 1:
        example_show_plot = False
    elif show_plot == False:
        example_show_plot = False
    else:
        example_show_plot = True

    # List of base seeds
    seeds = [base_seed + seed_offset * 10 for seed_offset in range(num_seeds)]
    
    # TODO figure out why i get this error: OSError: [WinError 1450] Insufficient system resources exist to complete the requested service (memory problem?)
    if num_cores > 1:
        results = Parallel(n_jobs=num_cores)(delayed(evaluate_model)(seed, num_input_units=num_input_units, hidden_units=hidden_units, num_classes=num_classes, 
                                     description=description, lr=lr, num_train_steps=num_train_steps, debug=debug, 
                                     show_plot=example_show_plot, png_save_path=png_save_path, svg_save_path=svg_save_path,
                                     test=test, plot_example_seed=base_seed, extra_params=extra_params, return_net=return_net) for seed in seeds)
    else:
        # Run without multiprocessing
        results = [evaluate_model(seed, num_input_units=num_input_units, hidden_units=hidden_units, num_classes=num_classes, 
                                     description=description, lr=lr, num_train_steps=num_train_steps, debug=debug, 
                                     show_plot=example_show_plot, png_save_path=png_save_path, svg_save_path=svg_save_path,
                                     test=test, plot_example_seed=base_seed, extra_params=extra_params, return_net=return_net) for seed in seeds]

    # Extract and average the metrics 
    val_accuracies = [result[1] for result in results]
    val_losses = [result[2] for result in results]
    test_accuracies = [result[3] for result in results]

    avg_val_acc = np.mean(val_accuracies)
    avg_val_loss = np.mean(val_losses)
    avg_test_acc = np.mean(test_accuracies) if test else None

    if num_seeds > 1 and verbose:
        print(f"Averaged Test Accuracy: {avg_test_acc:.3f}")
        print(f"Averaged Validation Accuracy: {avg_val_acc:.3f}")
        print(f"Averaged Validation Loss: {avg_val_loss:.3f}")
        sys.stdout.flush()

    model_dict = {description: {seed: results[i][0] for i, seed in enumerate(seeds)}}
    if export:
        os.makedirs(export_file_path, exist_ok=True)
        model_file_path = os.path.join(export_file_path, f"{description}_models.pkl")
        with open(model_file_path, "wb") as f:
            pickle.dump(model_dict, f)
        print(f"Network exported to {model_file_path}")
    
    if return_net:
        # Plotting
        if show_plot and test:
            idx = 0
            rep_net = results[0][idx]
            seed = seeds[idx]
            plot_title = label_dict[description]
            rep_net.display_summary(model_dict, title=plot_title, seed=seed, png_save_path=png_save_path, svg_save_path=svg_save_path, show_plot=show_plot)
            rep_net.plot_params(title=plot_title, seed=seed, png_save_path=png_save_path, svg_save_path=svg_save_path, show_plot=show_plot) 


    if return_net:
        return avg_val_acc, model_dict
    else:
        return avg_val_acc, None


@click.command()
@click.option('--description', required=True, type=str)
@click.option('--show_plot', is_flag=True, default=False) 
@click.option('--save_plot', is_flag=True, default=False)
@click.option('--interactive', is_flag=True)
@click.option('--export', is_flag=True, default=False)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='pkl_data')
@click.option('--seed', type=int, default=0)
@click.option('--debug', is_flag=True)
@click.option('--num_train_steps', type=int, default=None)
@click.option('--num_seeds', type=int, default=1)
@click.option('--num_cores', type=int, default=None)
def main(description, show_plot, save_plot, interactive, export, export_file_path, seed, debug, num_train_steps, num_seeds,
         num_cores):
    start_time = time.time()

    base_seed = seed

    num_input_units = 2 
    hidden_units = [128, 32]
    num_classes = 4 

    label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
                  'backprop_zero_bias': 'Backprop Zero Bias',
                  'backprop_fixed_bias': 'Backprop Fixed Bias',
                  'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast Learned Bias',
                  'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias',
                  'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias',
                  'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                  'ojas_dend_zero_bias': 'Oja\'s Rule Zero Bias',
                  'ojas_dend_fixed_bias': 'Oja\'s Rule Fixed Bias',
                  'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast Learned Bias',
                  'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                  'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}
        
    lr_dict = {'backprop_learned_bias': 0.1,
               'backprop_zero_bias': 0.01,
               'backprop_fixed_bias': 0.06,
               'dend_temp_contrast_learned_bias': 0.14,
               'dend_temp_contrast_zero_bias': 0.01,
               'dend_temp_contrast_fixed_bias': 0.07,
               'ojas_dend_learned_bias': 0.01,
               'ojas_dend_zero_bias': 0.02,
               'ojas_dend_fixed_bias':  0.0069,
               'dend_EI_contrast_learned_bias': 0.101,
               'dend_EI_contrast_zero_bias': 0.179,
               'dend_EI_contrast_fixed_bias': 0.04576}
    
    lr = lr_dict[description]
      
    extra_params = {}
    if "ojas_dend" in description:
        if "fixed_bias" in description:
            extra_params['alpha_Out'] = 0.0590
            extra_params['alpha_H2'] = 0.2274
            extra_params['alpha_H1'] = 1.2339
            # extra_params['alpha_Input'] = 0.8469
            extra_params['beta_Out'] = 1.8881
            extra_params['beta_H2'] =  1.2264
            extra_params['beta_H1'] = 1.7417
            # extra_params['beta_Input'] = 0.8497
            # extra_params['lr_Out'] = 0.0106
            # extra_params['lr_H2'] = 0.0106
            # extra_params['lr_H1'] = 0.0106
            # # extra_params['lr_Input'] = lr_dict[description]
        if "zero_bias" in description:
            extra_params['alpha'] = 0.6427
            extra_params['beta'] = 1.2165
    elif "dend_EI_contrast" in description:
        if "learned_bias" in description:
            extra_params['rec_lr_H1'] = 0.05
            extra_params['rec_lr_H2'] = 0.05
        elif "zero_bias" in description:
            extra_params['rec_lr_H1'] = 0.05
            extra_params['rec_lr_H2'] = 0.05
        elif "fixed_bias" in description:
            extra_params['rec_lr_H1'] = 0.88063
            extra_params['rec_lr_H2'] = 0.05593
        for i in range(len(hidden_units)):
            rec_layer_key = f'rec_lr_H{i+1}'
            if rec_layer_key not in extra_params:
                extra_params[rec_layer_key] = lr

    if save_plot:
        png_save_path = "figures"
        svg_save_path = "svg_figures"
        os.makedirs(png_save_path, exist_ok=True)
        os.makedirs(svg_save_path, exist_ok=True)
    else:
        png_save_path = None
        svg_save_path = None

    mean_val_accuracy, model_dict = eval_model_multiple_seeds(description, lr, base_seed, num_seeds, num_cores, num_input_units, hidden_units, num_classes,
                                                  export, export_file_path, show_plot, png_save_path, svg_save_path,
                                                  label_dict, debug, num_train_steps, test=True, extra_params=extra_params,
                                                  return_net=True)




    end_time = time.time()
    total_time = end_time - start_time
    if debug:
        print(f"Total execution time: {total_time:.3f} seconds")

    if interactive:
        globals().update(locals())


if __name__ == "__main__":
    main(standalone_mode=False)
