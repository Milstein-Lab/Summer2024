import pickle 
from spiral_neuralnet import *

@click.command()
@click.option('--description', required=True, type=str, default='backprop_learned_bias')
def main(description):
    seed = 2021
    data_split_seed = seed
    network_seed = seed + 1
    data_order_seed = seed + 2
    DEVICE = set_device()
    local_torch_random = torch.Generator()

    num_classes = 4
    X_test, y_test, X_train, y_train, test_loader, train_loader, data_fig = generate_data(K=num_classes, seed=data_split_seed, gen=local_torch_random, display=True)

    # Train and Test model
    set_seed(network_seed)
    local_torch_random.manual_seed(data_order_seed)

    label_dict = {'backprop_learned_bias': 'Backprop Learned Bias',
                'backprop_zero_bias': 'Backprop Zero Bias',
                'backprop_fixed_bias': 'Backprop Fixed Bias',
                'dend_temp_contrast_learned_bias': 'Dendritic Temporal Contrast Learned Bias',
                'dend_temp_contrast_zero_bias': 'Dendritic Temporal Contrast Zero Bias', 
                'dend_temp_contrast_fixed_bias': 'Dendritic Temporal Contrast Fixed Bias',
                'ojas_dend_learned_bias': 'Oja\'s Rule Learned Bias',
                'ojas_dend_zero_bias': 'Oja\'s Zero Bias',
                'ojas_dend_fixed_bias': 'Oja\'s Fixed Bias',
                'dend_EI_contrast_learned_bias': 'Dendritic EI Contrast Learned Bias',
                'dend_EI_contrast_zero_bias': 'Dendritic EI Contrast Zero Bias',
                'dend_EI_contrast_fixed_bias': 'Dendritic EI Contrast Fixed Bias'}

    with open("data/spiralNet_exported_model_data.pkl", "rb") as f:
        model_data_dict = pickle.load(f)

    net = model_data_dict[description]

    plot_title = label_dict[description]
    summary_fig = net.display_summary(test_loader, net.test_acc, title=plot_title, save_path=None, show_plot=False)
    params_fig = net.plot_params(title=plot_title, save_path=None, show_plot=False)

    plt.figure(data_fig.number)
    plt.figure(summary_fig.number)
    plt.figure(params_fig.number)
    plt.show() 

    globals().update(locals())


if __name__ == "__main__":
    main(standalone_mode=False)