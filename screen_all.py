import optuna
from spiral_neuralnet import *

start_time = time.time()

def objective(trial, config, base_seed, num_seeds=5, num_epochs=1):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.5, log=True)

    # Extract parameter ranges
    param_ranges = config.get("param_ranges", {})

    # Initialize extra_params with suggested values
    extra_params = {}
    for param, (low, high) in param_ranges.items():
        extra_params[param] = trial.suggest_float(param, low, high)

    val_accuracies = eval_model_multiple_seeds(
        lr=learning_rate, 
        base_seed=base_seed,
        num_seeds=num_seeds,
        extra_params=extra_params,
        test=False,
        verbose=False,
        return_net=False,
        **config
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
        "param_ranges": {},
        "num_cores": None,
        "label_dict": {}
    }

    # Populate param_ranges based on description
    if "ojas_dend" in description:
        config["param_ranges"] = {
            "alpha": (1e-5, 2),
            "beta": (1e-5, 2)
        }
    elif "dend_EI_contrast" in description:
        hidden_units = config['hidden_units']
        for i in range(len(hidden_units)):
            rec_layer_key = f'rec_lr_H{i+1}'
            config["param_ranges"][rec_layer_key] = (1e-5, 2)

    base_seed = 0

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, config, base_seed), n_trials=num_trials)

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Val Accuracy: {best_trial.value}")
    print("  Params: ")
    best_params_text = "Best Params:\n"
    for key, value in best_trial.params.items():
        print(f"    {key}: {value:.4f}")
        best_params_text += f"{key}: {value:.4f}\n"

    # Plotting
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(title_text="Accuracy over Trials", xaxis_title="Trial", yaxis_title='Accuracy')
    fig.add_annotation(text=best_params_text, xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False, align="left")
    fig.show()

    for param in study.best_params:
        fig = optuna.visualization.plot_slice(study, params=[param])
        fig.update_layout(title_text=f"{param} vs Accuracy", xaxis_title=f"{param} Value", yaxis_title='Accuracy')
        fig.add_annotation(text=best_params_text, xref="paper", yref="paper", x=0.5, y=-0.2, showarrow=False, align="left")
        fig.show()

if __name__ == "__main__":
    main(standalone_mode=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time: {total_time:.3f} seconds")