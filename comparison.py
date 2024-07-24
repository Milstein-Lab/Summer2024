from spiral_neuralnet import *
import spiral_neuralnet as spiral
from scipy.stats import ttest_ind

@click.command()
@click.option('--description1', type=str)
@click.option('--description2', type=str)
@click.option('--description3', type=str)
def main(description1, description2, description3):

    label_dict = {'backprop_learned_bias': 'Backprop\nLearned Bias',
                'backprop_zero_bias': 'Backprop Zero Bias',
                'backprop_fixed_bias': 'Backprop Fixed Bias',
                'dend_temp_contrast_learned_bias': 'Dendritic Temp Contrast\nLearned Bias',
                'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias',
                'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias',
                'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                'ojas_dend_zero_bias': 'Oja\'s Zero Bias',
                'ojas_dend_fixed_bias': 'Oja\'s Fixed Bias',
                'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast\nLearned Bias',
                'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}
    
    plt.rcParams.update({"axes.spines.right": False, 
                        "axes.spines.top": False,
                        "text.usetex": False, 
                         "font.size": 11,
                         "svg.fonttype": "none", 
                         "font.family": "Verdana"})
    
    descriptions = []
    if description1:
        descriptions.append(description1)
    if description2:
        descriptions.append(description2)
    if description3:
        descriptions.append(description3)
    if len(descriptions) == 0:
        descriptions = label_dict.keys()
    
    pkl_dir = 'pkl_data'
    pkl_files = []
    for d in descriptions:
        pkl_files.append(d + '_models.pkl')
    
    val_accuracies = {}
    individual_accuracies = {}

    for pkl in pkl_files:
        description = pkl.replace('_models.pkl', '')
        pkl_path = os.path.join(pkl_dir, pkl)

        with open(os.path.join(pkl_dir, pkl), 'rb') as f:
            dict = pickle.load(f)
        val_acc = dict['val_acc']
        nets_dict = dict[description]

        val_accuracies[description] = val_acc
        individual_accuracies[description] = [net.val_acc for net in nets_dict.values()]

    labels = [label_dict[key] for key in val_accuracies.keys()]
    accuracies = [val_accuracies[key] for key in val_accuracies.keys()]

    plt.figure(figsize=(10,6))
    bars = plt.bar(labels, accuracies)
    plt.xlabel('Model Variations')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Comparison of Neural Network Variations')
    plt.xticks(range(len(labels)))
    plt.ylim(70, 100)

    r = 45 if len(descriptions) > 3 else None
    h = 'right' if len(descriptions) > 3 else 'center'
    plt.xticks(labels, rotation=r, ha=h)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(i, 72, f'val acc: {round(yval, 2)}', va='center', ha='center', color='white', zorder=5) 

    for i, description in enumerate(val_accuracies.keys()):
        scatter_y = individual_accuracies[description]
        scatter_x = [i] * len(scatter_y)
        plt.scatter(scatter_x, scatter_y, color='red', zorder=5)

    # Perform t-tests
    reference_desc = descriptions[0]
    reference_accuracies = individual_accuracies[reference_desc]
    p_values = []

    for i, description in enumerate(descriptions[1:]):
        accuracies_to_compare = individual_accuracies[description]
        t_stat, p_val = ttest_ind(reference_accuracies, accuracies_to_compare)
        p_values.append(p_val)
        # Annotate the p-value on the graph
        plt.text(i+1, 75, f'p={p_val:.3f}', ha='center', va='bottom', color='black', fontsize=10)
    
    plt.tight_layout()

    save_path = 'svg_figures'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/comparison.svg', bbox_inches='tight', format='svg')

    plt.show()

if __name__ == "__main__":
    main(standalone_mode=False)