# Summer2024 - Spiral Dataset Classification Task

This repository contains a project that tests biologically plausible learning rules in Artificial Neural Networks (ANNs) using the spiral dataset classification task. The primary objective is to explore alternative learning mechanisms that more closely mimic the brain's learning processes and could enhance the efficiency of neural networks and help us understand how the brain learns.

## Research Focus

This project investigates the following learning rules:

* Backpropagation
* Dendritic Temporal Contrast
* Oja's Rule
* Dendritic EI (Excitatory-Inhibitory) Contrast

## Getting Started

### Prerequisites

* Python 3.11
* Create a virtual environment (optional but recommended):
```
conda create -n my_env python=3.11
```
* Install the required packages using:
```
pip install -r requirements.txt
```

### Running the Project

1. Clone the repository:
```
git clone https://github.com/Milstein-Lab/Summer2024.git
```

2. Navigate to the project directory:
```
cd Summer2024
```

3. Run the main script:
```
python -i spiral_neuralnet.py
```

Flags to add at the end of the command:
* ```--description```: (required) Set equal to the description of the model to run after this command
* ```--show_plot```: Displays plots in new window
* ```--save_plot```: Saves plots in the specified directory (by default will save to "/figures" and "/svg_figures"
* ```--export```: Exports network object to a pickle file
* ```--export_file_path```: Set equal to the file path to save pickle files to (by default will save to "/pkl_data")
* ```--seed```: Set equal to the desired random seed for the network and data generation
* ```--num_seeds```: Set equal to the number of seeds to test (by default will run 1 seed)
* ```--num_cores```: Set equal to the number of CPU cores to run code with (useful for testing multiple seeds)

