import optuna
import spiral_neuralnet as spiral
from spiral_neuralnet import *

def objective(trial, config, base_seed, num_seeds=5, num_epochs=1):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.5, log=True)

    # Extract parameter ranges
    param_ranges = config.get("extra_params", {})

    # Initialize extra_params with suggested values
    extra_params = {}
    for param, (low, high) in param_ranges.items():
        extra_params[param] = trial.suggest_float(param, low, high)

    val_accuracies = eval_model_multiple_seeds(
        description=config['description'], 
        lr=learning_rate, 
        base_seed=base_seed,
        num_seeds=num_seeds,
        num_input_units=config['num_input_units'],
        hidden_units=config['hidden_units'],
        num_classes=config['num_classes'],
        export=False,
        export_file_path=None,
        show_plot=False,
        png_save_path=None,
        svg_save_path=None,
        label_dict={},
        debug=False,
        num_train_steps=None,
        extra_params=extra_params,
        test=True,
        verbose=False
    )

    avg_accuracy = np.mean(val_accuracies)
    return avg_accuracy

@click.command()
@click.option('--description', required=True, type=str)
@click.option('--num_trials', type=int, default=40)
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data')
def main(description, num_trials, export, export_file_path):
    # Configuration dictionary
    config = {
        "description": description,
        "num_input_units": 2,
        "hidden_units": [128, 32],
        "num_classes": 4,
        "debug": False,
        "num_train_steps": None,
        "show_plot": False,
        "png_save_path": None,
        "svg_save_path": None,
        "export": False,
        "export_file_path": None,
        "extra_params": {}
    }

    # Populate extra_params based on description
    if "ojas_dend" in description:
        config["extra_params"] = {
            "alpha": (1e-5, 2),
            "beta": (1e-5, 2)
        }
    elif "dend_EI_contrast" in description:
        hidden_units = config['hidden_units']
        for i in range(len(hidden_units)):
            rec_layer_key = f'rec_lr_H{i+1}'
            config["extra_params"][rec_layer_key] = (1e-5, 2)

    base_seed = 0

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, base_seed), n_trials=num_trials)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Val Accuracy: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value:.4f}")

if __name__ == "__main__":
    main(standalone_mode=False)