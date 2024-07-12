from spiral_neuralnet import *

@click.command()
@click.option('--description', required=True, type=str)
def main(description):

    with open(f"pkl_data/screen_data_history.pkl", "rb") as f:
        screen_hist = pickle.load(f)

    print(screen_hist[description])

    with open("pkl_data/optuna_studies.pkl", "rb") as g:
        studies = pickle.load(g)
    
    print(studies[description])

if __name__ == "__main__":
    main(standalone_mode=False)