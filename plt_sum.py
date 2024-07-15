import os
from spiral_neuralnet import Net
import pickle
import numpy as np
import matplotlib.pyplot as plt
import click


def load_models(description, export_file_path):
    nets_file_path = os.path.join(export_file_path, f"{description}_models.pkl")

    if os.path.exists(nets_file_path):
        with open(nets_file_path, "rb") as f:  # 'rb' mode opens the file for reading in binary format
            model_dict = pickle.load(f)
        return model_dict
    else:
        print(f"No pickle file found at {nets_file_path}")
        return None


def plot_averaged_results(description, export_file_path):
    model_dict = load_models(description, export_file_path)
    if model_dict is None:
        return

    # Extract
    results = model_dict[description]

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

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps_list, average_train_accuracies, '-', label="Averaged Train Accuracy")
    plt.fill_between(train_steps_list, average_train_accuracies - std_train_accuracies,
                     average_train_accuracies + std_train_accuracies, alpha=0.2)

    # Colors!
    # seed_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    #                'tab:olive', 'tab:cyan']
    # for idx, (seed, result) in enumerate(results.items()):
    #     color = seed_colors[idx % len(seed_colors)]
    #     plt.plot(train_steps_list, result.train_accuracy, 'o', color=color, label=f'Seed {seed}', markersize=2)

    # Annotations/accuracy readings
    plt.annotate(f"Avg Val Acc: {average_val_acc:.2f}%", xy=(0.95, 0.9), xycoords='axes fraction', color='blue')
    plt.annotate(f"Avg Test Acc: {average_test_acc:.2f}%", xy=(0.95, 0.85), xycoords='axes fraction', color='orange')

    plt.xlabel('Train Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best', frameon=False)

    plt.title(f'Averaged Results for {description}')
    plt.tight_layout()
    plt.show(block=False)

    # Losses
    all_train_losses = [results[seed].avg_loss for seed in results]
    train_steps_list = results[next(iter(results))].train_steps_list  # Assuming all have the same train steps

    # Average losses and standard deviations
    average_train_losses = np.mean(all_train_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)

    # Plotting Accuracies
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_steps_list, average_train_accuracies, '-', label="Averaged Train Accuracy")
    plt.fill_between(train_steps_list, average_train_accuracies - std_train_accuracies,
                     average_train_accuracies + std_train_accuracies, alpha=0.2)

    # Colors
    # seed_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    #                'tab:olive', 'tab:cyan']
    # for idx, (seed, result) in enumerate(results.items()):
    #     color = seed_colors[idx % len(seed_colors)]
    #     plt.plot(train_steps_list, result.train_accuracy, 'o', color=color, label=f'Seed {seed}', markersize=2)
    #
    # plt.annotate(f"Avg Val Acc: {average_val_acc:.2f}%", xy=(0.95, 0.9), xycoords='axes fraction', color='blue')
    # plt.annotate(f"Avg Test Acc: {average_test_acc:.2f}%", xy=(0.95, 0.85), xycoords='axes fraction', color='orange')

    plt.xlabel('Train Steps')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best', frameon=False)
    plt.title(f'Averaged Results for {description}')

    # Plotting Losses
    plt.subplot(2, 1, 2)
    plt.plot(train_steps_list, average_train_losses, '-', label="Averaged Train Loss")
    plt.fill_between(train_steps_list, average_train_losses - std_train_losses, average_train_losses + std_train_losses,
                     alpha=0.2)

    # for idx, (seed, result) in enumerate(results.items()):
    #     color = seed_colors[idx % len(seed_colors)]
    #     plt.plot(train_steps_list, result.avg_loss, 'o', color=color, label=f'Seed {seed}', markersize=2)

    plt.xlabel('Train Steps')
    plt.ylabel('Loss')
    plt.legend(loc='best', frameon=False)

    plt.tight_layout()
    plt.show(block=False)
def plot_example_seed(description, example_seed, export_file_path):
    model_dict = load_models(description, export_file_path)
    if model_dict is None:
        return
    net = model_dict[description][example_seed]
    title = description
    net.plot_params(title=title, seed=example_seed, show_plot=True)
    net.display_summary(title = title, seed=example_seed, show_plot=True)


@click.command()
@click.option('--example_seed', required=True, type=int, help='Seed to build model', default=0)
@click.option('--description', required=True, type=str, help='Description of the model')
@click.option('--export_file_path', type=click.Path(file_okay=True), default='pkl_data',
              help='Path to the directory containing the exported model pickle files')
def main(description, example_seed, export_file_path):
    plot_averaged_results(description, export_file_path)
    plot_example_seed(description, example_seed, export_file_path)


if __name__ == "__main__":
    main(standalone_mode=False)
