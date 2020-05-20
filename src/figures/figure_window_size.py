import json
import matplotlib.pyplot as pl

markers = ['X', 'D', 'o']

SMALL_SIZE = 30
pl.rc('font', size=SMALL_SIZE)
pl.rc('axes', titlesize=SMALL_SIZE)
pl.rc('axes', labelsize=SMALL_SIZE)
pl.rc('xtick', labelsize=SMALL_SIZE)
pl.rc('ytick', labelsize=SMALL_SIZE)
pl.rc('legend', fontsize=SMALL_SIZE)
pl.rc('figure', titlesize=SMALL_SIZE+5)

small_dataset_path = '' # path to results of live test
big_dataset_path = '' # path to results of live video streaming event

algorithms = {
    'DVGAE': 'DVGAE/window_size',
    'EGCN': 'EGCN/window_size',
    'ABD': 'Dummy Attention/window_size',
}

options = ['1', '2', '3', '4']

files = {
    'mean_mse': 'mean mse.json',
    'var_mse': 'variance mse.json',
    'mean_mae': 'mean mae.json',
    'var_mae': 'variance mae.json',
    'mean_mse_thres': 'mean mse thres.json',
    'var_mse_thres': 'variance mse thres.json',
    'mean_mae_thres': 'mean mae thres.json',
    'var_mae_thres': 'variance mae thres.json',
}

mse_mean_files = ['mean_mse_thres']
mse_var_files = ['var_mse_thres']
mae_mean_files = ['mean_mae_thres']
mae_var_files = ['var_mae_thres']
labels = ['']


def read_list_from(json_path):
    with open(json_path, 'r') as handle:
        list = json.load(handle)
    return list


def load_data_from(path):
    result = {}

    for algorithm in algorithms.keys():
        result[algorithm] = {}
        alg_path = path + '/' + algorithms[algorithm]

        for option in options:
            result[algorithm][option] = {}
            exp_path = alg_path + '/' + option

            for file in files.keys():
                file_path = files[file]
                result[algorithm][option][file] = read_list_from(exp_path + '/' + file_path)

    return result


def get_value(a_list):
    return a_list[int(len(a_list) / 2) - 1]


def create_plot(data, mean_files, var_files, labels, ylabel, xlabel, title, plot_id):
    fig, ax = pl.subplots(figsize=(10, 10))
    for c, algorithm in enumerate(algorithms):
        for mean_file, var_file, label in zip(mean_files, var_files, labels, ):
            line = []
            errors = []
            for option in options:
                line.append(get_value(data[algorithm][option][mean_file]))
                errors.append(get_value(data[algorithm][option][var_file]))
            ax.plot(options, line, linewidth=5, label=algorithm + ' ' + label, marker=markers[c], markersize=20)
    ax.set_ylabel(ylabel, fontweight='semibold')
    ax.set_xlabel(xlabel, fontweight='semibold')
    ax.set_title(title)
    ax.legend(loc=10, bbox_to_anchor=(0.8, 0.7))
    fig.tight_layout()


def create_msae_plots(data, title, plot_id):
    create_plot(data, mse_mean_files, mse_var_files, labels, 'MSE', 'Window size', title, plot_id)
    create_plot(data, mae_mean_files, mae_var_files, labels, 'MAE', 'Window size', title, plot_id+1)


def plot_everything_from(path, label, plot_id):
    data = load_data_from(path)
    create_msae_plots(data, label, plot_id)


plot_everything_from(big_dataset_path, 'Live video streaming event', 0)
plot_everything_from(small_dataset_path, 'Live test', 2)
pl.show()


