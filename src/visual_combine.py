import matplotlib.gridspec as gridspec
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_comparison_from_files_with_padding(file_paths, metric_index, labels, title, save_path, fig_size=(4, 4), x_max=None, y_min=0, y_max=None, loc='upper right'):
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=fig_size)
    sns.set_theme(style="darkgrid")

    # Initialize a color palette
    # "hsv" is just an example, choose as per your preference
    palette = sns.color_palette("deep", len(file_paths))

    # Find the maximum length among all datasets to ensure uniform plotting
    max_length = 0
    avg_datasets = []
    all_datasets = []
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0]
            all_data = data[1]
            avg_datasets.append(avg_data[metric_index])
            all_datasets.append(all_data[metric_index])
            max_length = max(max_length, len(avg_data[metric_index]))

    # Plot each dataset, padding with NaNs where necessary
    for i, (avg_dataset, all_dataset, label) in enumerate(zip(avg_datasets, all_datasets, labels)):
        padded_avg_data = np.pad(
            avg_dataset, (0, max_length - len(avg_dataset)), 'constant', constant_values=np.nan)

        x_axis = np.arange(max_length)
        x_axis = x_axis if x_max == None else x_axis[:x_max]

        # Specify color from the palette
        color = palette[i]

        # Using pandas to handle NaNs gracefully in lineplot
        padded_avg_data = padded_avg_data if x_max == None else padded_avg_data[:x_max]
        df = pd.DataFrame(
            {'Epoch': x_axis, 'Value': padded_avg_data, 'Label': label})
        sns.lineplot(x='Epoch', y='Value', data=df,
                     label=label, linewidth=1, color=color)

        # Plot individual data points using Seaborn scatterplot for each data set
        for data_set in all_dataset:
            data_set = data_set if x_max == None else data_set[:x_max]
            sns.scatterplot(x=np.arange(len(data_set)),
                            y=data_set, alpha=0.1, s=20, color=color, legend=False)

    # Setting the y-axis limits
    plt.ylim(y_min, 1.0 if y_max == None else y_max)

    metric_name = "loss" if metric_index == 0 else "acc"
    # plt.title(f"{metric_name}")
    plt.xlabel(None)
    plt.ylabel(None)
    plt.legend(loc=loc)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_comparison_with_broken_y_axis_and_different_sizes(file_paths, metric_index, labels, title, save_path,
                                                           fig_size=(4, 4), x_max=None, y_min=0, y_max=None,
                                                           loc='upper right', break_point_start=None, break_point_end=None,
                                                           top_subplot_size_ratio=2, bottom_subplot_size_ratio=1):
    # Initialize the figure
    plt.figure(figsize=fig_size)
    sns.set_theme(style="darkgrid")

    # Create a gridspec with two rows of different heights
    gs = gridspec.GridSpec(2, 1, height_ratios=[
                           top_subplot_size_ratio, bottom_subplot_size_ratio])

    # Create two subplots with the specified size ratios
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    # Handling the palette
    palette = sns.color_palette("deep", len(file_paths))

    # Variables to store data for plotting
    max_length = 0
    avg_datasets = []
    all_datasets = []

    # Load data
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            avg_data = data[0][metric_index]
            all_data = [ds[:x_max] for ds in data[1][metric_index]]
            max_length = max(max_length, len(avg_data), *(len(ds)
                             for ds in all_data))
            avg_datasets.append(avg_data)
            all_datasets.append(all_data)

    # Determine x-axis range
    x_axis = np.arange(max_length)
    if x_max is not None:
        x_axis = x_axis[:x_max]

    for i, (avg_dataset, all_dataset, label) in enumerate(zip(avg_datasets, all_datasets, labels)):
        color = palette[i]

        # Adjust and plot data
        padded_avg_data = np.pad(avg_dataset, (0, max_length - len(avg_dataset)),
                                 'constant', constant_values=np.nan)[:x_max] if x_max else avg_dataset
        sns.lineplot(x=x_axis, y=padded_avg_data, label=label,
                     linewidth=1, color=color, ax=ax1, legend=False)
        sns.lineplot(x=x_axis, y=padded_avg_data, label=label,
                     linewidth=1, color=color, ax=ax2)
        for data_set in all_dataset:
            sns.scatterplot(x=np.arange(len(data_set)), y=data_set,
                            alpha=0.1, s=20, color=color, legend=False, ax=ax1)
            sns.scatterplot(x=np.arange(len(data_set)), y=data_set,
                            alpha=0.1, s=20, color=color, legend=False, ax=ax2)

    # Apply the break in y-axis between the specified points
    ax1.set_ylim(break_point_end, y_max if y_max is not None else 1.0)
    ax2.set_ylim(y_min, break_point_start)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.tick_bottom()
    ax1.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)

    # Broken axis marks
    d = .25  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    # Legend and titles
    ax2.legend(loc=loc)
    metric_name = "loss" if metric_index == 0 else "acc"
    # ax1.set_title(f"{metric_name} : {title}")
    # ax2.set_xlabel('Epoch')
    # ax1.set_ylabel(metric_name)
    # ax2.set_ylabel(metric_name)
    plt.subplots_adjust(hspace=0.0)  # Adjust space between subplots
    plt.tight_layout()

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, f"{title}_broken_y_axis.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


# TODO: expand 200 -> 1000
# The CIFAR-10 dataset
# consists of 60000 32x32 colour images in 10 classes,
# with 6000 images per class.
# There are 50000 training images and 10000 test images.
def expand_data_numpy(data, repeat):
    expanded_data = [np.repeat(sublist, repeat).tolist() for sublist in data]
    return expanded_data


if __name__ == '__main__':
    plot_directory = './save/avg_objects'
    save_path = './save/combined'
    metric_index = 1  # 0 for "loss", 1 for "acc"

    """
    1. Performance
    """
    multiplier = 5
    with open(f'{plot_directory}/nn_cifar_cnn_200_iid1.pkl', 'rb') as file:
        data = pickle.load(file)
        extended_avg_sgd = expand_data_numpy(data[0], multiplier)
        extended_all_sgd = [expand_data_numpy(d, multiplier) for d in data[1]]

    with open(f'{plot_directory}/nn_cifar_cnn_200_iid1_extended.pkl', 'wb') as f:
        pickle.dump([extended_avg_sgd, extended_all_sgd], f)

    title = 'Convergence'
    file_paths = [
        f'{plot_directory}/brain_cifar_cnn_1000_C0.1_iid1_E10_B50_Z0_SZ0_D0.55_W4_S4_TH0.0.pkl',
        f'{plot_directory}/fedasync_cifar_cnn_1000_C0.1_iid1_E10_B50_Z0_S4_A0.6.pkl',
        f'{plot_directory}/fedavg_cifar_cnn_1000_C0.1_iid1_E10_B50_Z0.pkl',
        f'{plot_directory}/nn_cifar_cnn_200_iid1_extended.pkl'
    ]
    labels = [
        'BRAIN',
        'FedAsync',
        'FedAvg',
        'SGD'
    ]
    plot_comparison_from_files_with_padding(
        file_paths, metric_index, labels, title, save_path,
        fig_size=(6, 4), x_max=1000, y_min=0.50, y_max=0.7)

    """
    2. Byzantine (x5)
    - BRAIN:    0 / 5 / 10 / 11 / 15
    - FedAsync: 0 / 5 / 10 / 11 / 15
    - FedAvg:   0 / 5 / 10 / 11 / 15
    """
    for b, th in zip([0, 5, 10, 11, 15], [0.0, 0.125, 0.125, 0.12, 0.125]):
        title = f'Byzantine_{b}'
        file_paths = [
            f'{plot_directory}/brain_cifar_cnn_1000_C0.1_iid1_E10_B50_Z{b}_SZ0_D0.55_W4_S4_TH{th}.pkl',
            f'{plot_directory}/fedasync_cifar_cnn_1000_C0.1_iid1_E10_B50_Z{b}_S4_A0.6.pkl',
            f'{plot_directory}/fedavg_cifar_cnn_1000_C0.1_iid1_E10_B50_Z{b}.pkl'
        ]
        labels = [
            'BRAIN',
            'FedAsync',
            'FedAvg'
        ]
        plot_comparison_with_broken_y_axis_and_different_sizes(
            file_paths, metric_index, labels, title, save_path,
            fig_size=(4, 4), x_max=400, y_min=0.075, y_max=0.625,
            loc='upper right', break_point_start=0.225, break_point_end=0.475,
            top_subplot_size_ratio=3, bottom_subplot_size_ratio=2)
