import subprocess
import multiprocessing
import click
import pickle
import os

def run_script(description, export, export_file_path):
    command = [
        "python", "screen.py",
        "--description", description,
        "--standalone"
    ]
    if export:
        command.append("--export")
        command.extend(["--export_file_path", export_file_path])

    subprocess.run(command)

@click.command()
@click.option('--description1', required=True, type=str)
@click.option('--description2', required=False, type=str)
@click.option('--description3', required=False, type=str)
@click.option('--export', is_flag=True)
@click.option('--export_file_path', type=click.Path(file_okay=True), default='screen_data')
def main(description1, description2, description3, export, export_file_path):
    descriptions = [description1]
    if description2:
        descriptions.append(description2)
    if description3:
        descriptions.append(description3)

    processes = []
    for description in descriptions:
        p = multiprocessing.Process(target=run_script, args=(description, export, export_file_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Merge results from all pickle files into screen_data_history.pkl
    screen_data_history = {}
    if os.path.exists('pkl_data/screen_data_history.pkl'):
        with open('pkl_data/screen_data_history.pkl', 'rb') as f:
            screen_data_history = pickle.load(f)

    for description in descriptions:
        result_file_path = f'pkl_data/screen_data_{description}.pkl'
        if os.path.exists(result_file_path):
            with open(result_file_path, 'rb') as f:
                data = pickle.load(f)
            screen_data_history[description] = data
            os.remove(result_file_path)

    # Save merged results to a single file
    with open('pkl_data/screen_data_history.pkl', 'wb') as f:
        pickle.dump(screen_data_history, f)
        print('Merged results saved to pkl_data/screen_data_history.pkl')

if __name__ == "__main__":
    main(standalone_mode=False)