import spiral_neuralnet as spiral
from spiral_neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import click

@click.command()
@click.option('--description', required=True, type=str, default='backprop_learned_bias')
@click.option('--seed', type=int, default=2021)
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data')
def main(description, seed, export, export_file_path):
	data_split_seed = seed
	network_seed = seed + 1
	data_order_seed = seed + 2
	DEVICE = spiral.set_device()
	local_torch_random = torch.Generator()
	
	label_dict = {'backprop_learned_bias': 'Backprop learned bias',
			   'backprop_zero_bias': 'Backprop zero bias',
			   'backprop_fixed_bias': 'Backprop fixed bias',
			   'dend_temp_contrast': 'Dendritic Temporal Contrast'}

	num_classes = 4
	X_test, y_test, X_train, y_train, test_loader, train_loader = spiral.generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

	loss_history = []
	learning_rates = np.arange(0.001, 0.2 + 0.001, 0.01)
	for i in learning_rates:
		spiral.set_seed(network_seed)

		if 'backprop' in description:
			if description == 'backprop_learned_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=True).to(DEVICE)
			elif description == 'backprop_zero_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False, learn_bias=False).to(DEVICE)
			elif description == 'backprop_fixed_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=False).to(DEVICE)

			criterion = "MSELoss"
			optimizer = optim.SGD(net.parameters(), lr=i)
			num_epochs = 2
			local_torch_random.manual_seed(data_order_seed)
			loss = net.train_model_backprop(criterion, optimizer, train_loader, num_epochs=num_epochs, verbose=True, device=DEVICE)

		elif description == "dend_temp_contrast":
			criterion = "MSELoss"
			net = Net(nn.ReLU,  X_train.shape[1], [128, 32], num_classes, description=description).to(DEVICE)
			loss = net.train_model_tempcontrast(criterion, i, train_loader, num_epochs=2, verbose=True, device=DEVICE)

		loss_history.append(loss)
		print(f'Learning Rate: {i}\n')
			
	fig = plt.figure()
	plt.plot(learning_rates, loss_history)
	for i, txt in enumerate(loss_history):
		plt.annotate(f'{learning_rates[i]:.2f}', (learning_rates[i], txt)) 
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy')
	plt.title(f'Learning Rate Screen for {label_dict[description]}')
	if export:
		plt.savefig(f'{export_file_path}/{description}_screen.svg', format='svg')
		plt.savefig(f'{export_file_path}/{description}_screen.png', format='png')
	fig.show()

	# chartLabels = ['Using Learned Biases', 'No Biases', 'Using Biases without Learning']
	# chartData = [test_acc, no_bias_test_acc, fixed_bias_test_acc]
	# fig = plt.figure()
	# bars = plt.bar(chartLabels, chartData, alpha=0.5)
	# plt.xlabel('Network Model')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy of the Network')
	# for bar in bars:
	# 	yval = bar.get_height()
	# 	plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom') 
	# fig.show()

	plt.show()

if __name__ == "__main__":
	main(standalone_mode=False)