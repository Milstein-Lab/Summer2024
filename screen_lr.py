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
                'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                'ojas_dend_zero_bias': 'Oja\'s Zero Bias',
                'ojas_dend_fixed_bias': 'Oja\'s Fixed Bias',
                'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast Learned Bias',
                'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}

    num_classes = 4
    X_test, y_test, X_train, y_train, test_loader, train_loader, data_fig = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False)

    accuracy_history = []
    start = 0.01
    end = 0.20
    step = 0.01
    learning_rates = np.arange(start, end + step, step)
    criterion = "MSELoss"
    num_epochs = 2
    for i in learning_rates:
        spiral.set_seed(network_seed)
        local_torch_random.manual_seed(data_order_seed)

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
                  learn_bias=learn_bias, mean_subtract_input=mean_subtract_input).to(DEVICE)

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
        os.makedirs('svg_figures', exist_ok=True)
        os.makedirs(export_file_path, exist_ok=True)
        plt.savefig(f'svg_figures/{description}_screen_{start}-{end}.svg', format='svg')
        plt.savefig(f'{export_file_path}/{description}_screen_{start}-{end}.png', format='png')
    fig.show()

    plt.show()

if __name__ == "__main__":
    main(standalone_mode=False)