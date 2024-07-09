import pickle
from spiral_neuralnet import *
import spiral_neuralnet as spiral
import torch
import matplotlib.pyplot as plt
import os

# No longer works with changes made to how models are pickled

def main():
    seed = 0
    data_split_seed = seed
    network_seed = seed + 1
    data_order_seed = seed + 2
    DEVICE = spiral.set_device()
    local_torch_random = torch.Generator()

    spiral.set_seed(network_seed)
    local_torch_random.manual_seed(data_order_seed)

    pkl_directory = "pkl_data"
    pkl_files = [f for f in os.listdir(pkl_directory) if f.endswith('.pkl')]

    num_classes = 4
    X_test, y_test, X_train, y_train, X_val, y_val, test_loader, train_loader, val_loader, _ = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

    label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
                'backprop_zero_bias': 'Backprop Zero Bias',
                'backprop_fixed_bias': 'Backprop Fixed Bias',
                'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast Learned Bias',
                'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias',
                'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias'}
                # 'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                # 'ojas_dend_zero_bias': 'Oja\'s Zero Bias',
                # 'ojas_dend_fixed_bias': 'Oja\'s Fixed Bias',
                # 'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast Learned Bias',
                # 'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                # 'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}
    
    test_accuracies = {}
    accuracy_histories = {}
    for pkl_file in pkl_files:
        description = pkl_file.replace('_model.pkl', '')
        pkl_path = os.path.join(pkl_directory, pkl_file)
        with open(pkl_path, "rb") as f:
            net = pickle.load(f)
        
        test_acc = net.test_model(test_loader, verbose=False, device=DEVICE)
        test_accuracies[description] = test_acc
        accuracy_histories[description] = net.averaged_accuracy

    labels = [label_dict[key] for key in test_accuracies.keys()]
    accuracies = [test_accuracies[key] for key in test_accuracies.keys()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accuracies, color='skyblue')
    ax.set_xlabel('Model Variations')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Comparison of Neural Network Variations')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center') 
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(10, 6))
    for description, accuracy in accuracy_histories.items():
        steps = list(range(len(accuracy)))
        ax.plot(steps, accuracy, label=label_dict[description])
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Averaged Accuracy')
    ax.set_title('Averaged Accuracy Over Training Steps for Different Models')
    ax.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()