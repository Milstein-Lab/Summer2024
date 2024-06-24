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


# set_seed() and seed_worker()
def set_seed(seed=None, seed_torch=True):
	"""
	Function that controls randomness. NumPy and random modules must be imported.
  
	Args:
	  seed : Integer
		A non-negative integer that defines the random state. Default is `None`.
	  seed_torch : Boolean
		If `True` sets the random seed for pytorch tensors, so pytorch module
		must be imported. Default is `True`.
  
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

	print(f'Random seed {seed} has been set.')

# In case that `DataLoader` is used
def seed_worker(worker_id):
	"""
	DataLoader will reseed workers following randomness in
	multi-process data loading algorithm.
  
	Args:
	  worker_id: integer
		ID of subprocess to seed. 0 means that
		the data will be loaded in the main process
		Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details
  
	Returns:
	  Nothing
	"""
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)
	
# set_device() to CPU or GPU
def set_device():
	"""
	Set the device. CUDA if available, CPU otherwise
  
	Args:
	  None
  
	Returns:
	  Nothing
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
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
	  K: int
		Number of classes
	  sigma: float
		Standard deviation
	  N: int
		Number of data points
  
	Returns:
	  X: torch.tensor
		Spiral data
	  y: torch.tensor
		Corresponding ground truth
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
		self.forward_activity = {} # activities of all layers post activation
		self.weights = {}
		self.biases = {}
		self.initial_weights = {}
		self.initial_biases = {}
		self.activation_functions = {}
		self.layers = {}

		self.hidden_activations = []

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
			self.initial_biases[key] = layer.bias.data.clone()

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
		self.initial_biases['Out'] = out_layer.bias.data.clone()

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
				self.forward_soma_state[key] = x.detach().clone()
			
			x = self.activation_functions[key](x)

			if store:
				self.forward_activity[key] = x.detach().clone()
		
		return x
	
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
	
	def train_backprop(self, optimizer, loss):
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	def train_model(self, learning_rule, lr, criterion, optimizer, train_loader, num_epochs=1, verbose=False, device='cpu'):
		"""
		Train model with backprop, accumulate loss, evaluate performance. 
	
		Args:
		- learning_rule (string): Learning rule to train model
		- lr (float): Learning rate
		- criterion (torch.nn type): Loss function
		- optimizer (torch.optim type): Implements Adam or MSELoss algorithm.
		- train_loader (torch.utils.data type): Combines the train dataset and sampler, and provides an iterable over the given dataset.
		- test_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset.
		- num_epochs (int): Number of epochs [default: 1]
		- verbose (boolean): If True, print statistics
		- device (string): CUDA/GPU if available, CPU otherwise
	
		Returns:
		- Nothing
		"""
		self.to(device)
		self.train()
		self.training_losses = []

		self.forward_soma_state_train_history = {}
		self.forward_activity_train_history = {} 
		self.forward_activity_train_history['Input'] = []
		for key, layer in self.layers.items():
			self.forward_soma_state_train_history[key] = []
			self.forward_activity_train_history[key] = []

		self.train_labels = []

		for epoch in tqdm(range(num_epochs)):  # Loop over the dataset multiple times
			for i, data in enumerate(train_loader, 0):
				# Get the inputs; data is a list of [inputs, labels]
				inputs, labels = data
				inputs = inputs.to(device).float()
				labels = labels.to(device).long()

				# forward + backward + optimize
				outputs = self.forward(inputs)
				self.forward_activity_train_history['Input'].append(self.forward_activity['Input'])
				for key, layer in self.layers.items():
					self.forward_soma_state_train_history[key].append(self.forward_soma_state[key])
					self.forward_activity_train_history[key].append(self.forward_activity[key])

				# Decide criterion function
				criterion_function = eval(f"nn.{criterion}()")
				if criterion == "MSELoss":
					targets = torch.zeros((inputs.shape[0],4))
					for row in range(len(labels)):
						col = labels.int()[row]
						targets[row][col] = 1
					loss = criterion_function(outputs, targets)
				elif criterion == "CrossEntropyLoss":
					loss = criterion_function(outputs, labels)
					
				self.train_labels.append(labels)

				if learning_rule == 'backprop':
					self.train_backprop(optimizer, loss)

				self.training_losses.append(loss.item())

		for key, layer in self.layers.items():
			self.forward_soma_state_train_history[key] = torch.stack(self.forward_soma_state_train_history[key]).squeeze()
			self.forward_activity_train_history[key] = torch.stack(self.forward_activity_train_history[key]).squeeze()
		self.train_labels = torch.stack(self.train_labels)
		
		self.eval()

		train_total, train_acc = self.test(train_loader, device)
		self.train_acc = train_acc

		self.final_weights = {}
		self.final_biases = {}
		for key, layer in self.layers.items():
			self.final_weights[key] = layer.weight.data.clone()
			self.final_biases[key] = layer.bias.data.clone()

		if verbose:
			print(f'\nAccuracy on the {train_total} training samples: {train_acc:0.2f}')

		return train_acc
	
	def ReLU_derivative(self, x):
		output = torch.ones_like(x)
		indexes = torch.where(output < 0)
		output[indexes] = 0
		return output
	
	def update_forward_states(self, outF):
		with torch.no_grad():
			Df_H2 = outF @ self.mlp[4].weight.data
			Df_H1 = self.hidden_outputs[-1] @ self.mlp[2].weight.data

		return Df_H2, Df_H1

	def backward_tempcontrast(self, nudge, Df_H1, Df_H2):
		with torch.no_grad():
			outB = self.mlp[-1](self.output_state + nudge)
			Db_H2 = outB @ self.mlp[4].weight.data
			nudge_H2 = Db_H2 - Df_H2
			activity_b_H2 = self.mlp[-3](self.hidden_states[1] + nudge_H2)
			Db_H1 = activity_b_H2 @ self.mlp[2].weight.data
			nudge_H1 = Db_H1 - Df_H1
			activity_b_H1 = self.mlp[1](self.hidden_states[0] + nudge_H1)
		
		return nudge_H1, nudge_H2, activity_b_H1, activity_b_H2

	def step_tempcontrast(self, lr, inputs, nudge_H1, nudge_H2, nudge_out):
		with torch.no_grad():
			# print(f"output_state post activation size: {self.ReLU_derivative(self.hidden_states[0]).size()}")
			# print(f"nudge_H2 size: {nudge_H2.size()}")
			# print(f"hidden_outputs[1] size: {self.hidden_outputs[0].size()}")
			# print(f"outer prodct size: {torch.outer(nudge_H2.squeeze(), self.hidden_outputs[0].squeeze()) .size()}")

			self.mlp[4].weight.data += lr * self.ReLU_derivative(self.output_state) @ torch.outer(nudge_out.squeeze(), self.hidden_outputs[1].squeeze())
			self.mlp[2].weight.data += -lr * self.ReLU_derivative(self.hidden_states[1]) @ torch.outer(nudge_H2.squeeze(), self.hidden_outputs[0].squeeze()) 
			self.mlp[0].weight.data += -lr * self.ReLU_derivative(self.hidden_states[0]) @ torch.outer(nudge_H1.squeeze(), inputs.squeeze())

	def train_model_tempcontrast(self, criterion, lr, train_loader, num_epochs=1, verbose=False, device='cpu'):
			'''
			Train model using Target propogation Temporal contrast learning

			Args:
			- lr (float): Learning rate 
			- train_loader (torch.utils.data.DataLoader): DataLoader object containing the training data.
			- num_epochs (int, optional): Number of training epochs. Defaults to 1.
			- verbose (bool, optional): If True, prints the accuracy on the training samples. Defaults to False.
			- device (str, optional): Device to use for training. Defaults to 'cpu'.

			Returns:
			- float: Accuracy on the training samples.
			'''	
			self.to(device)
			self.eval()
			debug = False

			self.training_losses = []
			self.reinit()

			self.train_hidden_states = [[] for i in range(len(self.hidden_unit_nums))]
			self.train_hidden_outputs = [[] for i in range(len(self.hidden_unit_nums))]
			self.train_output_state = []
			self.train_processed_output = []

			self.train_labels = []

			for epoch in tqdm(range(num_epochs)): 
				for i, data in enumerate(train_loader, 0):
					inputs, labels = data
					inputs = inputs.to(device).float()
					labels = labels.to(device).long()

					outF = self.forward(inputs)

					for layer_idx in range(len(self.hidden_unit_nums)):
						self.train_hidden_states[layer_idx].append(self.hidden_states[layer_idx])
						self.train_hidden_outputs[layer_idx].append(self.hidden_outputs[layer_idx])
					self.train_output_state.append(self.output_state)
					self.train_processed_output.append(self.processed_output)

					if debug:
						if i % 100 == 0:
							print(f"Epoch {epoch}, Batch {i}, Forward output: {outF.mean().item()}")

					# update forward (pre nudge) states (to get Df)
					Df_H2, Df_H1 = self.update_forward_states(outF)

					# compute nudge
					targets = torch.zeros((inputs.shape[0],4))
					for row in range(len(labels)):
						col = labels.int()[row]
						targets[row][col] = 1
					nudge_out = targets - outF

					if debug:
						if i % 100 == 0:
							print(f"Epoch {epoch}, Batch {i}, Nudge: {nudge_out.mean().item()}")

					# update backward (post nudge) states (to get Db)
					nudge_H1, nudge_H2, activity_b_H1, activity_b_H2 = self.backward_tempcontrast(nudge_out, Df_H1, Df_H2)
					
					# step (based on Df and Db)
					self.step_tempcontrast(lr, inputs, nudge_H1, nudge_H2, nudge_out)

					# Decide criterion function
					criterion_function = eval(f"nn.{criterion}()")
					if criterion == "MSELoss":
						loss = criterion_function(outF, targets)
					elif criterion == "CrossEntropyLoss":
						loss = criterion_function(outF, labels)

					self.train_labels.append(labels)

					self.training_losses.append(loss.item())

			# Concatenate hidden states and outputs within the current batch
			for layer_idx in range(len(self.hidden_unit_nums)):
				self.train_hidden_states[layer_idx] = torch.stack(self.train_hidden_states[layer_idx]).squeeze()
				self.train_hidden_outputs[layer_idx] = torch.stack(self.train_hidden_outputs[layer_idx]).squeeze()
			self.train_output_state = torch.stack(self.train_output_state).squeeze()
			self.train_processed_output = torch.stack(self.train_processed_output).squeeze()
			self.train_labels = torch.stack(self.train_labels)

			train_total, train_acc = self.test(train_loader, device)

			self.final_weights = []
			self.final_biases = []
			for layer in self.mlp[::2]:
				self.final_weights.append(layer.weight.data)
				if self.use_bias:
					self.final_biases.append(layer.bias.data)

			if verbose:
				print(f'\nAccuracy on the {train_total} training samples: {train_acc:0.2f}')

			return train_acc
	
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
		DEVICE: cpu or gpu
			Device type
		X_test: torch.tensor
			Test data
		y_test: torch.tensor
			Labels of the test data
		M: int
			Size of the constructed tensor with meshgrid
		x_max: float
			Defines range for the set of points
		eps: float
			Decision threshold
	
		Returns:
		Nothing
		"""
		X_all = sample_grid()
		y_pred = self.forward(X_all.to(DEVICE), store=False).cpu()

		decision_map = torch.argmax(y_pred, dim=1)

		for i in range(len(X_test)):
			indices = (X_all[:, 0] - X_test[i, 0])**2 + (X_all[:, 1] - X_test[i, 1])**2 < eps
			decision_map[indices] = (K + y_test[i]).long()

		decision_map = decision_map.view(M, M)

		return decision_map.T

	def display_summary(self, test_loader, test_acc, title=None):
		'''
		Display network summary

		Args:
		- test_loader (torch.utils.data type): Combines the test dataset and sampler, and provides an iterable over the given dataset.
		- device (string): CUDA/GPU if available, CPU otherwise

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
		fig.show()

	def plot_params(self, title=None):
		'''
		Plot initial and final weights and biases for all layers.

		Args:
		- Nothing

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
		axes[0].scatter(X[:, 0], X[:, 1], c = y)
		axes[0].set_xlabel('x1')
		axes[0].set_ylabel('x2')
		axes[0].set_title('Train Data')

		axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
		axes[1].set_xlabel('x1')
		axes[1].set_ylabel('x2')
		axes[1].set_title('Test data')

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
def main(description, plot, interactive, export, export_file_path, seed):
	data_split_seed = seed
	network_seed = seed + 1
	data_order_seed = seed + 2
	DEVICE = set_device()
	local_torch_random = torch.Generator()

	num_classes = 4
	X_test, y_test, X_train, y_train, test_loader, train_loader = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=plot)

	# Train and Test model
	set_seed(network_seed)

	label_dict = {'backprop_learned_bias': 'Backprop learned bias',
			   'backprop_zero_bias': 'Backprop zero bias',
			   'backprop_fixed_bias': 'Backprop fixed bias',
				'dend_temp_contrast': 'Dendritic Temporal Contrast'}
	
	lr_dict = {'backprop_learned_bias': 0.11,
			   'backprop_zero_bias': 0.01,
			   'backprop_fixed_bias': 0.10,
			   'dend_temp_contrast': 0.01}
	
	if "backprop" in description: 
		if description == 'backprop_learned_bias':
			net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=True).to(DEVICE)
		elif description == 'backprop_zero_bias':
			net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False, learn_bias=False).to(DEVICE)
		elif description == 'backprop_fixed_bias':
			net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=False).to(DEVICE)

		criterion = "MSELoss"
		optimizer = optim.SGD(net.parameters(), lr=lr_dict[description])
		num_epochs = 2
		local_torch_random.manual_seed(data_order_seed)
		net.train_model('backprop', lr_dict[description], criterion, optimizer, train_loader, num_epochs=num_epochs, device=DEVICE)
		test_acc = net.test_model(test_loader, verbose=False, device=DEVICE)

	elif description == "dend_temp_contrast":
		criterion = "MSELoss"
		net = Net(nn.ReLU,  X_train.shape[1], [128, 32], num_classes, description=description).to(DEVICE)
		net.train_model_tempcontrast(criterion, lr_dict[description], train_loader, num_epochs=2, verbose=True, device=DEVICE)
		test_acc = net.test_model(test_loader, verbose=True, device=DEVICE)

	if plot:
		net.display_summary(test_loader, test_acc, title=label_dict[description])
		net.plot_params(title=label_dict[description])

	if export:
		if os.path.isfile(export_file_path):
			with open(export_file_path, "rb") as f:
				model_data_dict = pickle.load(f)
		else:
			model_data_dict = {}
		model_data_dict[description] = net
		with open(export_file_path, "wb") as f:
			pickle.dump(model_data_dict, f)

	if plot:
		plt.show()

	if interactive:
		globals().update(locals())


if __name__ == "__main__":
	main(standalone_mode=False)