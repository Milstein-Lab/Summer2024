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

	label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
				  'backprop_zero_bias': 'Backprop Zero Bias',
				  'backprop_fixed_bias': 'Backprop Fixed Bias',
				  'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast Learned Bias',
				  'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias',
				  'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias',
				  'ojas_learned_bias': 'Ojas Rule Learned Bias',
				  'ojas_zero_bias': 'Ojas Zero Bias',
				  'ojas_fixed_bias': 'Ojas Fixed Bias', }

	num_classes = 4
	X_test, y_test, X_train, y_train, test_loader, train_loader = spiral.generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

	accuracy_history = []
	start = 0.01
	end = 0.2
	step = 0.01
	learning_rates = np.arange(start, end + step, step)
	criterion = "MSELoss"
	num_epochs = 2
	for i in learning_rates:
		spiral.set_seed(network_seed)
		local_torch_random.manual_seed(data_order_seed)

		if "backprop" in description: 
			if description == 'backprop_learned_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=True).to(DEVICE)
			elif description == 'backprop_zero_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False, learn_bias=False).to(DEVICE)
			elif description == 'backprop_fixed_bias':
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=False).to(DEVICE)

		elif "dend_temp_contrast" in description:
			if description == "dend_temp_contrast_learned_bias":
				net = Net(nn.ReLU,  X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=True).to(DEVICE)
			elif description == "dend_temp_contrast_zero_bias":
				net = Net(nn.ReLU,  X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False, learn_bias=False).to(DEVICE)
			elif description == "dend_temp_contrast_fixed_bias":
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True, learn_bias=False).to(DEVICE)
		elif "ojas" in description:
			if description == "ojas_learned_bias":
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True,learn_bias=True).to(DEVICE)
			elif description == "ojas_zero_bias":
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=False,learn_bias=False).to(DEVICE)
			elif description == "ojas_fixed_bias":
				net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=True,learn_bias=False).to(DEVICE)

		acc = net.train_model(description, i, criterion, train_loader, debug=False, num_epochs=num_epochs, verbose=True, device=DEVICE)

		accuracy_history.append(acc)
		print(f'Learning Rate: {i}\n')
			

	max_idx = np.argmax(accuracy_history)
	max_accuracy = accuracy_history[max_idx]
	best_learning_rate = learning_rates[max_idx]
	textstr = f'Best Learning Rate: {best_learning_rate:.3f}\nTrain Accuracy: {max_accuracy:.3f}'

	fig = plt.figure()
	plt.plot(learning_rates, accuracy_history, label=textstr)
	plt.legend(handlelength=0)
	plt.scatter(best_learning_rate, max_accuracy, color='red')

	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy')
	plt.title(f'Learning Rate Screen for {label_dict[description]}')
	if export:
		plt.savefig(f'{export_file_path}/{description}_screen_{start}-{end}.svg', format='svg')
		plt.savefig(f'{export_file_path}/{description}_screen_{start}-{end}.png', format='png')
	fig.show()

	plt.show()

if __name__ == "__main__":
	main(standalone_mode=False)