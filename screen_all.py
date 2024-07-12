import optuna
import plotly.graph_objs as go
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
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data_history')
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

    best_params_dict = {}

    print(f"  Val Accuracy: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value:.4f}")
        best_params_dict[key] = value

    # Plotting
    graphs_dir = 'screen_data'
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs('svg_figures', exist_ok=True)

    fig = go.Figure()
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(
        title_text=f"{description} Accuracy over Trials",
        xaxis_title="Trial",
        yaxis_title='Accuracy'
    )
    best_params_text = "\n".join([f"Best {key}: {value:.4f}" for key, value in best_params_dict.items()])
    fig.add_annotation(
        text=best_params_text,
        xref="paper",
        yref="paper",
        x=0,
        y=1,
        showarrow=False,
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    fig.write_image(f"{graphs_dir}/{description}_screen.png")
    fig.write_image(f"svg_figures/{description}_screen.svg")
    fig.show()

    for param in study.best_params:
        fig = go.Figure()
        fig = optuna.visualization.plot_slice(study, params=[param])
        fig.update_layout(
            title_text=f"{description} {param} vs Accuracy",
            xaxis_title=f"{param} Value",
            yaxis_title='Accuracy'
        )
        
        # Determine position based on the highest accuracy value
        trials_df = study.trials_dataframe()
        max_accuracy = trials_df['value'].max()
        max_accuracy_index = trials_df['value'].idxmax()
        best_param_value = trials_df.at[max_accuracy_index, f'params_{param}']
        
        fig.add_annotation(
            text=f"Best {param}: {best_param_value:.4f}",
            x=best_param_value,
            y=max_accuracy,
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=1,
            arrowsize=0.5,
            arrowwidth=1,
            arrowcolor="black",
            ax=-40,
            ay=-40,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8,
            align="left"
        )
        fig.write_image(f"{graphs_dir}/{description}_{param}_screen.png")
        fig.write_image(f"svg_figures/{description}_{param}_screen.svg")
        fig.show()

    if export:
        screen_data_dict = {'learning_rates': [], 'accuracies': [], 'extra_params': {}}
        for trial_num, trial in enumerate(study.trials):
            screen_data_dict['learning_rates'].append(trial.params['learning_rate'])
            screen_data_dict['accuracies'].append(trial.value)

            extra_params_for_trial = {key: value for key, value in trial.params.items() if key != 'learning_rate'}
            if extra_params_for_trial:
                screen_data_dict['extra_params'][f"trial_{trial_num}"] = extra_params_for_trial

        if os.path.exists(f'pkl_data/{export_file_path}.pkl'):
            with open(f'pkl_data/{export_file_path}.pkl', 'rb') as f:
                screen_data_history = pickle.load(f)
        else:
            screen_data_history = {}
        screen_data_history[description] = screen_data_dict
        os.makedirs('pkl_data', exist_ok=True)
        with open(f'pkl_data/{export_file_path}.pkl', 'wb') as f:
            pickle.dump(screen_data_history, f)
        print(f'Screen data history saved to pkl_data/{export_file_path}')

        if os.path.exists('pkl_data/optuna_studies.pkl'):
            with open(f'pkl_data/optuna_studies.pkl', 'rb') as f:
                optuna_studies = pickle.load(f)
        else:
            optuna_studies = {}
        optuna_studies[description] = study
        with open('pkl_data/optuna_studies.pkl', 'wb') as f:
            pickle.dump(optuna_studies, f)
        print('Optuna study saved to pkl_data/optuna_studies')


if __name__ == "__main__":
    main(standalone_mode=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.3f} seconds")