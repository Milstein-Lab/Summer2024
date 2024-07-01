# Imports
import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from IPython.display import display
import random
import pickle
import os
import click
import traceback


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

    def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num, description=None, use_bias=True, learn_bias=True):
        """
        Initialize MLP Network parameters
    
        Args:
        - actv (string): Activation function
        - input_feature_num (int): Number of input features
        - hidden_unit_nums (list): Number of units per hidden layer. List of integers
        - output_feature_num (int): Number of output features
        - description (string): Learning rule to use
        - use_bias (boolean): If True, randomly initialize biases. If False, set all biases to 0.
        - learn_bias (boolean): If True, use learning rule to update biases. If False, otherwise.
    
        Returns:
        - Nothing
        """
        super(Net, self).__init__()
        # Save parameters as self variables
        self.input_feature_num = input_feature_num
        self.hidden_unit_nums = hidden_unit_nums
        self.output_feature_num = output_feature_num
        self.description = description
        self.use_bias = use_bias
        self.learn_bias = learn_bias

        self.forward_soma_state = {} # states of all layers pre activation
        self.forward_activity = {} # activities of all layers post ReLU
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

        if 'dend_EI_contrast' in self.description:
            self.recurrent_layers = {}
            self.recurrent_weights = {}
            for i, num in enumerate(self.hidden_unit_nums):
                layer = f'H{i+1}'
                self.recurrent_layers[layer] = nn.Linear(num, num, bias=False)
                self.recurrent_weights[layer] = self.recurrent_layers[layer].weight

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

    def forward(self, x, store=True):
        """
        Simulate forward pass of MLP Network
    
        Args:
        - x (torch.tensor): Input data
        - store (boolean): If True, store intermediate states and activities of each layer
    
        Returns:
        - x (torch.tensor): Output data
        """
        if store:
            self.forward_activity['Input'] = x.detach().clone()

        for key, layer in self.layers.items():
            x = layer(x)
            if store:
                self.forward_soma_state[key] = x.detach().clone() # Before ReLU
            
            x = self.activation_functions[key](x)

            if store:
                self.forward_activity[key] = x.detach().clone() # After ReLU
        
        return x

    def save_gradients(self, key):
        def hook_fn(module, grad_input, grad_output):
            self.hooked_grads[key] = grad_output[0]
        return hook_fn

    def register_hooks(self):
        for key in self.layers.keys():
            self.layers[key].register_full_backward_hook(self.save_gradients(key))
    
    def test(self, data_loader, device='cpu'):
        """
        Function to gauge network performance
    
        Args:
        - data_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset.
        - device (string): CUDA/GPU if available, CPU otherwise
    
        Returns:
        - acc (float): Performance of the network
        - total (int): Number of datapoints in the dataloader
        """
        correct = 0
        total = 0
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        return total, acc
    
    def train_model(self, description, lr, criterion, train_loader, debug=False, num_train_steps=None, num_epochs=1, verbose=False, device='cpu'):
        """
        Train model with backprop, accumulate loss, evaluate performance.
    
        Args:
        - lr (float): Learning rate
        - criterion (torch.nn type): Loss function
        - optimizer (torch.optim type): Implements Adam or MSELoss algorithm.
        - train_loader (torch.utils.data type): Combines the train dataset and sampler, and provides an iterable over the given dataset.
        - debug (boolean): If True, enters debug mode.
        - num_train_steps (int): Stops train loop after specified number of steps.
        - num_epochs (int): Number of epochs [default: 1]
        - verbose (boolean): If True, print statistics
        - device (string): CUDA/GPU if available, CPU otherwise
    
        Returns:
        - Nothing
        """
        self.to(device)
        self.train()
        self.training_losses = []
        train_steps = 0

        # Create dictionaries for train state and activities
        self.forward_soma_state_train_history = {}
        self.forward_activity_train_history = {}
        self.forward_dend_state_train_history = {}
        self.backward_dend_state_train_history = {}
        self.nudges_train_history = {}
        self.weights_train_history = {}
        self.forward_activity_train_history['Input'] = []
        for key, layer in self.layers.items():
            self.forward_soma_state_train_history[key] = []
            self.forward_activity_train_history[key] = []
            self.forward_dend_state_train_history[key] = []
            self.backward_dend_state_train_history[key] = []
            self.nudges_train_history[key] = []
            self.weights_train_history[key] = []

        self.train_labels = []

        for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
            for i, data in enumerate(train_loader, 0):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                # forward pass
                outputs = self.forward(inputs)

                # Store forward state and activity info
                self.forward_activity_train_history['Input'].append(self.forward_activity['Input'])
                for key, layer in self.layers.items():
                    self.forward_soma_state_train_history[key].append(self.forward_soma_state[key])
                    self.forward_activity_train_history[key].append(self.forward_activity[key])

                # Decide criterion function
                criterion_function = eval(f"nn.{criterion}()")
                if criterion == "MSELoss":
                    targets = torch.zeros((inputs.shape[0], self.output_feature_num))
                    for row in range(len(labels)):
                        col = labels[row].int()
                        targets[row][col] = 1
                    loss = criterion_function(outputs, targets)
                elif criterion == "CrossEntropyLoss":
                    loss = criterion_function(outputs, labels)
                    
                self.train_labels.append(labels)

                # Choose learning rule
                if 'backprop' in description:
                    self.train_backprop(loss, lr)
                elif 'dend' in description:
                    self.train_dend(description, targets, lr)

                # store a copy of the weights in a weight_history dict
                for key, layer in self.weights.items():
                    self.weights_train_history[key].append(self.weights[key])

                # Track losses
                self.training_losses.append(loss.item())

                # Stop after one certain number of train step
                train_steps += 1
                if debug and num_train_steps is not None:
                    if train_steps == num_train_steps:
                        assert False

        # Squeeze all history tensors
        for key, layer in self.layers.items():
            self.forward_soma_state_train_history[key] = torch.stack(self.forward_soma_state_train_history[key]).squeeze()
            self.forward_activity_train_history[key] = torch.stack(self.forward_activity_train_history[key]).squeeze()
            if self.forward_dend_state_train_history[key]:
                self.forward_dend_state_train_history[key] = torch.stack(self.forward_dend_state_train_history[key]).squeeze()
            if self.backward_dend_state_train_history[key]:
                self.backward_dend_state_train_history[key] = torch.stack(self.backward_dend_state_train_history[key]).squeeze()
            if self.nudges_train_history[key]:
                self.nudges_train_history[key] = torch.stack(self.nudges_train_history[key]).squeeze()
            if self.weights_train_history[key]:
                self.weights_train_history[key] = torch.stack(self.weights_train_history[key]).squeeze()

        self.train_labels = torch.stack(self.train_labels)
        
        self.eval()

        train_total, train_acc = self.test(train_loader, device)
        self.train_acc = train_acc

        # Store final weights and biases
        self.final_weights = {}
        self.final_biases = {}
        for key, layer in self.layers.items():
            self.final_weights[key] = layer.weight.data.clone()
            self.final_biases[key] = self.biases[key].data.clone()

        if verbose:
            print(f'\nAccuracy on the {train_total} training samples: {train_acc:0.2f}')

        return train_acc
    
    def train_backprop(self, loss, lr):
        optimizer = optim.SGD(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def ReLU_derivative(self, x):
        output = torch.ones_like(x)
        indexes = torch.where(x <= 0)
        output[indexes] = 0
        return output

    def backward_dend_temp_contrast(self, targets):
        prev_layer = None
        reverse_layers = list(self.layers.keys())[::-1]
        
        for layer in reverse_layers:
            if layer == 'Out':
                self.nudges[layer] = (2.0 / self.output_feature_num) * self.ReLU_derivative(
                    self.forward_soma_state[layer]) * (targets - self.forward_activity[
                    'Out'])  # the ReLU derivative term is dA/dz
                self.nudges_train_history[layer].append(self.nudges[layer])
            else:
                self.forward_dend_state[layer] = self.forward_activity[prev_layer] @ self.weights[prev_layer]
                self.backward_dend_state[layer] = self.backward_activity[prev_layer] @ self.weights[prev_layer]
                self.nudges[layer] = self.ReLU_derivative(self.forward_soma_state[layer]) * (
                        self.backward_dend_state[layer] - self.forward_dend_state[layer])
                self.forward_dend_state_train_history[layer].append(self.nudges[layer])
                self.backward_dend_state_train_history[layer].append(self.backward_dend_state[layer])
                self.nudges_train_history[layer].append(self.nudges[layer])
            
            self.backward_activity[layer] = self.activation_functions[layer](
                self.forward_soma_state[layer] + self.nudges[layer])
            prev_layer = layer

    def backward_ojas(self, targets):
        prev_layer = None
        reverse_layers = list(self.layers.keys())[::-1]
        
        for layer in reverse_layers:
            if layer == 'Out':
                self.nudges[layer] = self.ReLU_derivative(self.forward_soma_state[layer]) * (targets - self.forward_activity['Out'])  # the ReLU derivative term is dA/dz
                self.nudges_train_history[layer].append(self.nudges[layer])
            else:
                self.forward_dend_state[layer] = self.forward_activity[prev_layer] @ self.weights[prev_layer]
                self.backward_dend_state[layer] = self.backward_activity[prev_layer] @ self.weights[prev_layer]
                self.nudges[layer] = self.ReLU_derivative(self.forward_soma_state[layer]) * (
                        self.backward_dend_state[layer] - self.forward_dend_state[layer])
                self.forward_dend_state_train_history[layer].append(self.nudges[layer])
                self.backward_dend_state_train_history[layer].append(self.backward_dend_state[layer])
                self.nudges_train_history[layer].append(self.nudges[layer])
            
            self.backward_activity[layer] = self.activation_functions[layer](
                self.forward_soma_state[layer] + self.nudges[layer])
            prev_layer = layer
    
    def step_dend_temp_contrast(self, lr):
        # add beta scalars?
        with torch.no_grad():
            prev_layer = 'Input'
            for layer in self.layers.keys():
                self.weights[layer].data += lr * torch.outer(self.nudges[layer].squeeze(),
                                                             self.forward_activity[prev_layer].squeeze())
                if self.use_bias and self.learn_bias:
                    self.biases[layer].data += lr * self.nudges[layer].squeeze()
                prev_layer = layer
    
    def step_ojas(self, lr):
        with ((torch.no_grad())):
            prev_layer = 'Input'
            for layer in self.layers.keys():
                self.weights[layer].data += (lr * self.backward_activity[layer].T * (self.forward_activity[prev_layer] - self.backward_activity[layer].T * self.weights[layer].data))
                # self.weights[layer].data += (lr * self.backward_activity[layer].T * (np.sign(self.weights[layer].data) * self.forward_activity[prev_layer] - self.backward_activity[layer].T * self.weights[layer].data))
                if self.use_bias and self.learn_bias:
                    self.biases[layer].data += lr * self.nudges[layer].squeeze()
                prev_layer = layer
    
    def backward_dend_EI_contrast(self, targets):
        reverse_layers = list(self.layers.keys())[::-1]
        
        for idx, layer in enumerate(reverse_layers):
            if layer == 'Out':
                self.nudges[layer] = (2.0 / self.output_feature_num) * self.ReLU_derivative(
                    self.forward_soma_state[layer]) * (targets - self.forward_activity['Out'])
            else:
                upper_layer = reverse_layers[idx - 1]
                self.forward_dend_state[layer] = self.forward_activity[upper_layer] @ self.weights[upper_layer] + \
                                                 self.recurrent_layers[layer](self.forward_activity[layer])
                self.backward_dend_state[layer] = self.backward_activity[upper_layer] @ self.weights[upper_layer] + \
                                                  self.recurrent_layers[layer](self.forward_activity[layer])
                self.nudges[layer] = self.backward_dend_state[layer] * self.ReLU_derivative(
                    self.forward_soma_state[layer])
            self.backward_activity[layer] = self.activation_functions[layer](
                self.forward_soma_state[layer] + self.nudges[layer])
    
    def step_dend_EI_contrast(self, lr):
        with torch.no_grad():
            lower_layer = 'Input'
            for layer in self.layers.keys():
                self.weights[layer].data += lr * torch.outer(self.nudges[layer].squeeze(),
                                                             self.forward_activity[lower_layer].squeeze())
                if layer != 'Out':
                    self.recurrent_weights[layer].data += -1 * lr * self.forward_dend_state[layer].T @ \
                                                          self.forward_activity[layer]
                if self.use_bias and self.learn_bias:
                    self.biases[layer].data += lr * self.nudges[layer].squeeze()
                lower_layer = layer
    
    def train_dend(self, description, targets, lr):
        '''
        Wrapper function for training models with dendritic learning rules
        - Dendritic Temporal Contrast
        - Oja's Rule
        - Dendritic Excitatory-Inhibitory (EI) Contrast

        Args:
        - targets (torch tensor): Target activities for output neurons
        - lr (float): Learning rate

        Returns:
        - Nothing
        '''
        self.eval()
        if 'dend_temp_contrast' in description:
            self.backward_dend_temp_contrast(targets)
            self.step_dend_temp_contrast(lr)
        elif 'oja' in description:
            self.backward_ojas(targets)
            self.step_ojas(lr)
        elif 'dend_EI_contrast' in description:
            self.backward_dend_EI_contrast(targets)
            self.step_dend_EI_contrast(lr)
    
    def test_model(self, test_loader, verbose=True, device='cpu'):
        '''
        Evaluate performance

        Args:
        - test_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset.
        - verbose (boolean): If True, print statistics
        - device (string): CUDA/GPU if available, CPU otherwise

        Returns:
        - Nothing
        '''
        self.to(device)
        test_total, test_acc = self.test(test_loader, device)

        if verbose:
            print(f'Accuracy on the {test_total} testing samples: {test_acc:0.2f}\n')
        
        self.test_acc = test_acc
        return test_acc
    
    def get_decision_map(self, X_test, y_test, K, DEVICE='cpu', M=500, eps=1e-3):
        """
        Helper function to plot decision map
    
        Args:
        - X_test (torch.tensor): Test data
        - y_test (torch.tensor): Labels of the test data
        - DEVICE (cpu or gpu): Device type
        - M (int): Size of the constructed tensor with meshgrid
        - eps (float): Decision threshold
    
        Returns:
        - Nothing
        """
        X_all = sample_grid()
        y_pred = self.forward(X_all.to(DEVICE), store=False).cpu()

        decision_map = torch.argmax(y_pred, dim=1)

        for i in range(len(X_test)):
            indices = (X_all[:, 0] - X_test[i, 0])**2 + (X_all[:, 1] - X_test[i, 1])**2 < eps
            decision_map[indices] = (K + y_test[i]).long()

        decision_map = decision_map.view(M, M)

        return decision_map.T

    def display_summary(self, test_loader, test_acc, title=None, save_path=None):
        '''
        Display network summary

        Args:
        - test_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset.
        - test_acc (int): Accuracy of model after testing.
        - title (string): Title of model based on description.

        Returns:
        - Nothing
        '''

        inputs, labels = next(iter(test_loader))

        class_averaged_activity = {}
        sorted_indices_layers = {}
        for key, activity in self.forward_activity.items():
            this_class_averaged_activity = torch.empty((self.output_feature_num, self.forward_activity[key].shape[1]))
            for label in torch.arange(self.output_feature_num):
                indexes = torch.where(labels == label)
                this_class_averaged_activity[label,:] = torch.mean(activity[indexes], dim=0)

            max_indices = this_class_averaged_activity.argmax(dim=0)
            if key == 'Out':
                this_sorted_indices = torch.arange(self.output_feature_num)
            else:
                this_sorted_indices = max_indices.argsort()

            class_averaged_activity[key] = this_class_averaged_activity
            sorted_indices_layers[key] = this_sorted_indices

        num_layers = len(self.hidden_unit_nums)+1
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

        axes[1][0].plot(self.training_losses, label=f"Test Accuracy: {test_acc}")
        axes[1][0].set_xlabel('Train Steps')
        axes[1][0].set_ylabel('Training Loss')
        axes[1][0].legend(loc='best', frameon=False)

        map = self.get_decision_map(inputs, labels, self.output_feature_num)

        axes[1][1].imshow(map, extent=[-2, 2, -2, 2], cmap='jet', origin='lower')
        axes[1][1].set_xlabel('x1')
        axes[1][1].set_ylabel('x2')
        axes[1][1].set_title('Predictions')

        for j in range(2,num_layers):
            axes[1][j].axis('off')

        if title is not None:
            fig.suptitle(f'Class Averaged Activity - {title}')
        else:
            fig.suptitle('Class Averaged Activity')

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(f'{save_path}/summary.png', bbox_inches='tight')
        else:
            fig.show()

    def plot_params(self, title=None, save_path=None):
        '''
        Plot initial and final weights and biases for all layers.

        Args:
        - title (string): Title of model based on description.

        Returns:
        - Nothing
        '''

        num_layers = len(self.layers)
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
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(f'{save_path}/parameters.png', bbox_inches='tight')
        else:
            fig.show()


def sample_grid(M=500, x_max=2.0):
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


def generate_data(K=4, sigma=0.16, N=1000, seed=None, gen=None, display=True):
    '''
    Generate spiral dataset for training and testing a neural network.

    Args:
    - K (int): Number of classes in the dataset. Default is 4.
    - sigma (float): Standard deviation of the spiral dataset. Default is 0.16.
    - N (int): Number of samples in the dataset. Default is 1000.
    - seed (int): Seed value for reproducibility. Default is None.
    - gen (torch.Generator): Generator object for random number generation. Default is None.
    - display (bool): Whether to display a scatter plot of the dataset. Default is True.

    Returns:
    - X_test (torch.Tensor): Test input data.
    - y_test (torch.Tensor): Test target data.
    - X_train (torch.Tensor): Train input data.
    - y_train (torch.Tensor): Train target data.
    - test_loader (torch.utils.data.DataLoader): DataLoader for test data.
    - train_loader (torch.utils.data.DataLoader): DataLoader for train data.
    '''

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    
    # Spiral Data Set graph
    X, y = create_spiral_dataset(K, sigma, N)
    
    num_samples = X.shape[0]
    # Shuffle data
    shuffled_indices = torch.randperm(num_samples)   # Get indices to shuffle data, could use torch.randperm
    X = X[shuffled_indices]
    y = y[shuffled_indices]

    # Split data into train/test
    test_size = int(0.2 * num_samples)    # Assign test datset size using 20% of samples
    X_test = X[:test_size]
    y_test = y[:test_size]
    X_train = X[test_size:]
    y_train = y[test_size:]

    if display:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes[0].scatter(X[:, 0], X[:, 1], c = y, s=10)
        axes[0].set_xlabel('x1')
        axes[0].set_ylabel('x2')
        axes[0].set_title('Train Data')

        axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10)
        axes[1].set_xlabel('x1')
        axes[1].set_ylabel('x2')
        axes[1].set_title('Test Data')

        fig.tight_layout()
        fig.show()

    # Train and test DataLoaders
    if gen is None:
        gen = torch.Generator()
    batch_size = 1
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=0,
                            worker_init_fn=seed_worker, generator=gen)
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0,
                            worker_init_fn=seed_worker, generator=gen)
    
    return X_test, y_test, X_train, y_train, test_loader, train_loader


@click.command()
@click.option('--description', required=True, type=str, default='backprop_learned_bias')
@click.option('--plot', is_flag=True)
@click.option('--interactive', is_flag=True)
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='data/spiralNet_exported_model_data.pkl')
@click.option('--seed', type=int, default=2021)
@click.option('--debug', is_flag=True)
@click.option('--num_train_steps', type=int, default=1)
def main(description, plot, interactive, export, export_file_path, seed, debug, num_train_steps):
    data_split_seed = seed
    network_seed = seed + 1
    data_order_seed = seed + 2
    DEVICE = set_device()
    local_torch_random = torch.Generator()

    num_classes = 4
    X_test, y_test, X_train, y_train, test_loader, train_loader = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

    def train_and_handle_debug(net, description, lr, criterion, train_loader, debug, num_train_steps, num_epochs, device):
        try:
            net.train_model(description, lr, criterion, train_loader, debug=debug, num_train_steps=num_train_steps, num_epochs=num_epochs, device=device)
        except AssertionError:
            print(f"{num_train_steps} train steps completed.")
        except Exception as e:
            traceback.print_exc()

    # Train and Test model
    set_seed(network_seed)

    label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
                'backprop_zero_bias': 'Backprop Zero Bias',
                'backprop_fixed_bias': 'Backprop Fixed Bias',
                'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast Learned Bias',
                'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias',
                'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias',
                'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                'ojas_dend_zero_bias': 'Oja\'s Zero Bias',
                'ojas_dend_fixed_bias': 'Oja\'s Fixed Bias',
                'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast Learned Bias',
                'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}
    
    lr_dict = {'backprop_learned_bias': 0.11,
               'backprop_zero_bias': 0.01,
               'backprop_fixed_bias': 0.08,
               'dend_temp_contrast_learned_bias': 0.13,
               'dend_temp_contrast_zero_bias': 0.01,
               'dend_temp_contrast_fixed_bias': 0.10,
               'ojas_dend_learned_bias': 0.13,
               'ojas_dend_zero_bias': 0.01,
               'ojas_dend_fixed_bias': 0.021,
               'dend_EI_contrast_learned_bias': 0.11,
               'dend_EI_contrast_zero_bias': 0.01,
               'dend_EI_contrast_fixed_bias': 0.031}
    
    criterion = "MSELoss"
    num_epochs = 2
    local_torch_random.manual_seed(data_order_seed)

    if "learned_bias" in description:
        net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=True).to(DEVICE)
    elif "zero_bias" in description:
        net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False, learn_bias=False).to(DEVICE)
    elif "fixed_bias" in description:
        net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=False).to(DEVICE)

    if debug:
        net.register_hooks()
        train_and_handle_debug(net, description, lr_dict[description], criterion, train_loader, debug, num_train_steps, num_epochs, DEVICE)
    else:
        net.train_model(description, lr_dict[description], criterion, train_loader, debug=debug, num_train_steps=num_train_steps, num_epochs=num_epochs, device=DEVICE)
        test_acc = net.test_model(test_loader, verbose=False, device=DEVICE)

        if plot:
            net.display_summary(test_loader, test_acc,  title=label_dict[description], save_path="figures")
            net.plot_params(title=label_dict[description], save_path="figures")

    if export:
        if os.path.isfile(export_file_path):
            with open(export_file_path, "rb") as f:
                model_data_dict = pickle.load(f)
        else:
            model_data_dict = {}
        model_data_dict[description] = net
        with open(export_file_path, "wb") as f:
            pickle.dump(model_data_dict, f)
        print(f"Network exported to {export_file_path}")

    if interactive:
        globals().update(locals())


if __name__ == "__main__":
    main(standalone_mode=False)