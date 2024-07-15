import optuna
import plotly.graph_objs as go
import sqlite3
from spiral_neuralnet import *

start_time = time.time()

def objective(trial, config, base_seed, db_path):
    trial_start_time = time.time()

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.5, log=True)

    # Extract parameter ranges
    param_ranges = config.get("param_ranges", {})

    # Initialize extra_params with suggested values
    extra_params = {}
    for param, (low, high) in param_ranges.items():
        extra_params[param] = trial.suggest_float(param, low, high)

    val_accuracies, _ = eval_model_multiple_seeds(
        lr=learning_rate, 
        base_seed=base_seed,
        extra_params=extra_params,
        test=False,
        verbose=False,
        return_net=False,
        **config
    )

    avg_accuracy = np.mean(val_accuracies)

    trial_end_time = time.time()
    trial_time = trial_end_time - trial_start_time

    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()

        # Get the current best trial number
        c.execute("SELECT trial_number, accuracy FROM trial_results ORDER BY accuracy DESC LIMIT 1")
        best_trial_row = c.fetchone()
        if best_trial_row is None or avg_accuracy > best_trial_row[1]:
            best_trial_so_far = trial.number
        else:
            best_trial_so_far = best_trial_row[0]

        c.execute(
            "INSERT INTO trial_results (trial_number, learning_rate, accuracy, extra_params, trial_time, best_trial_so_far) VALUES (?, ?, ?, ?, ?, ?)",
            (trial.number, learning_rate, avg_accuracy, str(extra_params), trial_time, best_trial_so_far)
        )
        conn.commit()

    return avg_accuracy

@click.command()
@click.option('--description', required=True, type=str)
@click.option('--num_trials', type=int, default=40)
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data_history')
@click.option('--num_seeds', type=int, default=5)
@click.option('--num_cores', type=int, default=None)
def main(description, num_trials, export, export_file_path, num_seeds, num_cores):
    
    if num_cores is None:
        num_cores = min(cpu_count(), num_seeds)
    else:
        num_cores = min(num_cores, num_seeds)

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
        "num_seeds": num_seeds,
        "num_cores": num_cores,
        "label_dict": {}
    }

    # Populate param_ranges based on description
    if "ojas_dend" in description:
        config["param_ranges"] = {
            "alpha_Out": (1e-5, 2),
            "alpha_H2":(1e-5, 2),
            "alpha_H1":(1e-5, 2),
            "beta_Out": (1e-5, 2),
            "beta_H2":(1e-5, 2),
            "beta_H1":(1e-5, 2),
        }
    elif "dend_EI_contrast" in description:
        hidden_units = config['hidden_units']
        for i in range(len(hidden_units)):
            rec_layer_key = f'rec_lr_H{i+1}'
            config["param_ranges"][rec_layer_key] = (1e-5, 2)

    base_seed = 0

    os.makedirs("screen_data", exist_ok=True)
    db_path = f"screen_data/{description}_optimization_results.db"
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS trial_results")
        c.execute("""
            CREATE TABLE IF NOT EXISTS trial_results (
                trial_number INTEGER PRIMARY KEY,
                learning_rate REAL,
                accuracy REAL,
                extra_params TEXT,
                trial_time REAL, 
                best_trial_so_far INTEGER
            )
        """)
        conn.commit()

    study = optuna.create_study(study_name=f'{description}_Optimization', direction="maximize")
    study.optimize(lambda trial: objective(trial, config, base_seed, db_path), n_trials=num_trials)

    print("Best trial:")
    best_trial = study.best_trial

    best_params_dict = {}

    print(f"  Val Accuracy: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
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
    # fig.write_image(f"{graphs_dir}/{description}_screen.png")
    # fig.write_image(f"svg_figures/{description}_screen.svg")
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
        # fig.write_image(f"{graphs_dir}/{description}_{param}_screen.png")
        # fig.write_image(f"svg_figures/{description}_{param}_screen.svg")
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

        study_path = 'optuna_studies'
        if os.path.exists(f'pkl_data/{study_path}.pkl'):
            with open(f'pkl_data/{study_path}.pkl', 'rb') as f:
                optuna_studies = pickle.load(f)
        else:
            optuna_studies = {}
        optuna_studies[description] = study
        with open(f'pkl_data/{study_path}.pkl', 'wb') as f:
            pickle.dump(optuna_studies, f)
        print(f'Optuna study saved to pkl_data/{study_path}')


if __name__ == "__main__":
    main(standalone_mode=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time:.3f} seconds")