import spiral_neuralnet as spiral
from spiral_neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def main():
	SEED = 2021
	data_split_seed = SEED
	network_seed = SEED + 1
	data_order_seed = SEED + 2
	DEVICE = spiral.set_device()
	local_torch_random = torch.Generator()
	
	K, X_test, y_test, X_train, y_train, test_loader, train_loader = spiral.generate_data(data_split_seed, local_torch_random, display=False)

	criterion = "MSELoss"
	num_epochs = 2

	loss_history = []
	learning_rates = np.arange(0.04, 0.15, 0.01)  # np.arange(0.02, 0.3 + 0.01, 0.01)
	for i in learning_rates:
		spiral.set_seed(network_seed)
		local_torch_random.manual_seed(data_order_seed)
		net = spiral.Net(nn.ReLU, X_train.shape[1], [128, 32], K, use_bias=True, learn_bias=False).to(DEVICE)
		optimizer = optim.SGD(net.parameters(), lr=i)
		loss = net.train_model(criterion, optimizer, train_loader, num_epochs=num_epochs, verbose=True, device=DEVICE)
		loss_history.append(loss)

	fig = plt.figure()
	plt.plot(learning_rates, loss_history)
	for i, txt in enumerate(loss_history):
		plt.annotate(f'{learning_rates[i]:.2f}', (learning_rates[i], txt)) 
	fig.show()

	plt.show()

if __name__ == "__main__":
	main()