import pickle
from spiral_neuralnet import *
import spiral_neuralnet as spiral
import torch
import matplotlib.pyplot as plt

def main():
	seed = 2021
	data_split_seed = seed
	network_seed = seed + 1
	data_order_seed = seed + 2
	DEVICE = spiral.set_device()
	local_torch_random = torch.Generator()

	spiral.set_seed(network_seed)
	local_torch_random.manual_seed(data_order_seed)

	with open("C:/Yash Dev/Summer2024/Yash/data/spiralNet_exported_model_data.pkl", "rb") as f:
		model_data_dict = pickle.load(f)

	num_classes = 4
	X_test, y_test, X_train, y_train, test_loader, train_loader = spiral.generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

	label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
				'backprop_zero_bias': 'Backprop Zero Bias',
			   	'backprop_fixed_bias': 'Backprop Fixed Bias',
				'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast\nLearned Bias',
			   	'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast\nZero Bias', 
			   	'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast\nFixed Bias'}
	
	test_accuracies = {}
	for description, net in model_data_dict.items():
		test_acc = net.test_model(test_loader, verbose=False, device=DEVICE)
		test_accuracies[description] = test_acc

	labels = [label_dict[key] for key in test_accuracies.keys()]
	accuracies = [test_accuracies[key] for key in test_accuracies.keys()]

	fig = plt.figure(figsize=(10, 6))
	bars = plt.bar(labels, accuracies, color='skyblue')
	plt.xlabel('Model Variations')
	plt.ylabel('Test Accuracy')
	plt.title('Comparison of Neural Network Variations')
	plt.xticks(rotation=45, ha='right')
	for bar in bars:
		yval = bar.get_height()
		plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center') 
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()