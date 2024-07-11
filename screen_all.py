import optuna
import numpy as np
import torch
from joblib import Parallel, delayed
import os
from spiral_neuralnet import *

def objective(trial, config, base_seed, num_seeds=5, num_epochs=1):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.5, log=True)
    
    # Extract parameter ranges
    param_ranges = config.get("extra_params", {})

    # Initialize extra_params with suggested values
    extra_params = {}
    for param, (low, high) in param_ranges.items():
        extra_params[param] = trial.suggest_float(param, low, high)

    def train_single_seed(seed_offset):
        num_classes = 4

        data_split_seed = 0 
        network_seed = base_seed + seed_offset * 10 + 1
        data_order_seed = base_seed + seed_offset * 10 + 2
        DEVICE = set_device()
        local_torch_random = torch.Generator()
        local_torch_random.manual_seed(data_order_seed)

        _, _, X_train, _, _, _, test_loader, train_loader, val_loader = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=False, png_save_path=None, svg_save_path=None)
        num_input_units = X_train.shape[1]

        description = config["description"]
        mean_subtract_input = "ojas_dend" in description
        use_bias = "learned_bias" in description or "fixed_bias" in description
        learn_bias = "learned_bias" in description
        use_bias = use_bias and not "zero_bias" in description

        net = Net(nn.ReLU, num_input_units, [128, 32], num_classes, description=description, use_bias=use_bias, learn_bias=learn_bias, 
              lr=learning_rate, extra_params=extra_params, mean_subtract_input=mean_subtract_input, seed=network_seed).to(DEVICE)

        val_acc = net.train_model(description, train_loader, val_loader, debug=False, num_train_steps=None, num_epochs=num_epochs, device=DEVICE)
        return val_acc

    # Parallelize over seeds
    num_cores = os.cpu_count()
    accuracy_list = Parallel(n_jobs=num_cores)(delayed(train_single_seed)(seed_offset) for seed_offset in range(num_seeds))
    
    avg_accuracy = np.mean(accuracy_list)
    return avg_accuracy

# Configuration dictionary
config = {
    "description": "ojas_dend_fixed_bias",
    "extra_params": {
        "alpha_out": (1e-5, 2),
        "alpha_h2": (1e-5, 2),
        "alpha_h1": (1e-5, 2),
        "alpha_Input": (1e-5, 2),
        "beta_out": (1e-5, 2),
        "beta_h2": (1e-5, 2),
        "beta_h1": (1e-5, 2),
        "beta_input": (1e-5, 2),
    }
}
base_seed = 0

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, config, base_seed), n_trials=400)

print("Best trial:")
trial = study.best_trial

print(f"  Val Accuracy: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value:.4f}")
