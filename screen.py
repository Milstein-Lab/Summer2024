import optuna
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import click
import pickle
import os
import spiral_neuralnet as spiral
from spiral_neuralnet import *

local_torch_random = torch.Generator()

def objective(trial, description, base_seed):
    # Parameters to be optimized
    learning_rate = trial.suggest_float('learning_rate', start, end, step=step)
    num_seeds = 5
    num_epochs = 1
    accuracy_list = []

    for seed_offset in range(num_seeds):
        # Set seeds
        data_split_seed = 0  # Fixed seed for data split
        network_seed = base_seed + seed_offset * 10 + 1
        data_order_seed = base_seed + seed_offset * 10 + 2

        # Data generation
        spiral.set_seed(data_split_seed)
        local_torch_random.manual_seed(data_order_seed)
        _, _, X_train, _, _, _, _, train_loader, val_loader, _ = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)
        
        # Network initialization
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

        net = Net(nn.ReLU, X_train.shape[1], [128, 32], num_classes, description=description, use_bias=use_bias,
                  learn_bias=learn_bias, lr=learning_rate, mean_subtract_input=mean_subtract_input).to(DEVICE)

        # Training
        acc = net.train_model(description, learning_rate, criterion, train_loader, val_loader, debug=False, num_epochs=num_epochs, verbose=False, device=DEVICE)
        accuracy_list.append(acc)

    avg_accuracy = np.mean(accuracy_list)
    return avg_accuracy

@click.command()
@click.option('--description', required=True, type=str, default='backprop_learned_bias')
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data')
def main(description, export, export_file_path):
    global num_classes, criterion, start, end, step, DEVICE
    base_seed = 0
    num_classes = 4
    start = 0.01
    end = 0.3
    step = 0.01
    criterion = "MSELoss"
    DEVICE = spiral.set_device()

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
    
    # Load existing data if available
    screen_data_dict = {}   
    if os.path.exists('pkl_data/screen_data_history.pkl'):
        with open('pkl_data/screen_data_history.pkl', 'rb') as f:
            screen_data_dict = pickle.load(f)
    
    # Run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, description, base_seed), n_trials=20)

    # Get the best learning rate
    best_learning_rate = study.best_params['learning_rate']
    best_accuracy = study.best_value

    # Extract all trials for plotting
    trials = study.trials
    learning_rates = [trial.params['learning_rate'] for trial in trials]
    accuracies = [trial.value for trial in trials]

    # Update the data dictionary
    screen_data_dict[description] = {
        'learning_rates': learning_rates,
        'accuracies': accuracies
    }

    # Save the updated dictionary
    os.makedirs('pkl_data', exist_ok=True)
    with open('pkl_data/screen_data_history.pkl', 'wb') as f:
        pickle.dump(screen_data_dict, f)

    # Plot results
    best_idx = np.argmax(accuracies)
    best_lr = learning_rates[best_idx]
    best_acc = accuracies[best_idx]

    textstr = f'Best Learning Rate: {best_lr:.3f}\nAverage Validation Accuracy: {best_acc:.3f}'

    fig = plt.figure()
    plt.scatter(learning_rates, accuracies, label=textstr)
    plt.legend(handlelength=0)
    plt.scatter(best_lr, best_acc, color='red')

    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Rate Screen for {label_dict[description]}')
    if export:
        os.makedirs('svg_figures', exist_ok=True)
        os.makedirs(export_file_path, exist_ok=True)
        plt.savefig(f'svg_figures/{description}_screen.svg', format='svg')
        plt.savefig(f'{export_file_path}/{description}_screen.png', format='png')
    fig.show()

    plt.show()

if __name__ == "__main__":
    main(standalone_mode=False)
