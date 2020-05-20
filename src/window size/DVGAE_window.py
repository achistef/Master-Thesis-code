# disable GPU acceleration
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import packages
import json
import numpy as np
import networkx as nx
import tensorflow as tf
from pathlib import Path
import scipy.sparse as sps
from collections import defaultdict
from statistics import mean, pvariance

# input/output directory
snapshot_prefix = ''  # the input directory should point to the graph snapshots directory
output_dir = 'results/DVGAE/window_size/'

# == parameters ==
# threshold for high bandwidth test edges
config_test_weight_threshold = 0.4
# first train snapshot
start_subgraph = 10
# last train snapshot
end_subgraph = 20
# test snapshot
test_subgraph = 299
num_snapshots = end_subgraph - start_subgraph
# intermediate embedding size
intermediate_size = 32
# embedding size 
embedding_size = 16
# number of train iterations
epochs = 100
# learning rate of training process
learning_rate = 0.001
# depth options of training (window)
l_depth_options = [1, 2, 3, 4]
# number of times the experiment is repeated
repeat = 5


# deconstructs a sparse matrix to coordinates, values and shape
def sparse_to_tuple(sparse_mx):
    sparse_mx = sps.triu(sparse_mx)
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# calculates the normalized laplacian of the input matrix
def calc_normalized(adj_):
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sps.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo().astype(np.float32)
    return adj_normalized


# calculates the mean square error
def calc_mse(coords, labels, embeddings):
    predictions = []

    for src, dst in coords:
        emb1 = embeddings[src]
        emb2 = embeddings[dst]
        pred = tf.sigmoid(tf.tensordot(emb1, emb2, 1)).numpy()
        predictions.append(pred)

    mse = tf.keras.losses.MSE(labels, predictions)

    return mse


# calculates the mean absolute error
def calc_mae(coords, labels, embeddings):
    predictions = []

    for src, dst in coords:
        emb1 = embeddings[src]
        emb2 = embeddings[dst]
        pred = tf.sigmoid(tf.tensordot(emb1, emb2, 1)).numpy().tolist()
        predictions.append(pred)

    predictions = tf.Variable(predictions)
    mae = tf.reduce_mean(tf.abs(labels - predictions))
    return mae


# removes nodes that do not have any high bandwidth connection
def clear_low_nodes(graph):
    delete = []
    for node in graph.nodes():
        neighbors = dict(graph.adj[node])
        vals = list(neighbors.values())
        weights = [d['weight'] for d in vals]
        filter = [weight > 0.5 for weight in weights]
        if not any(filter):
            delete.append(node)
    for node in delete:
        graph.remove_node(node)


# removes nodes that do not have any connections
def clear_empty_nodes(graph):
    delete = []
    for node in graph.nodes():
        neighbors = dict(graph.adj[node])
        vals = list(neighbors.values())
        weights = sum([d['weight'] for d in vals])
        if weights == 0:
            delete.append(node)
    for node in delete:
        graph.remove_node(node)


# the following lists hold the results of preprocessing

# adjacency matrix of each snapshot
adj_snapshots = []
# normalized adjacency matrix of each snapshot
adj_norm_snapshots = []
# node features per snapshot
features_snapshots = []
# number of nodes each snapshot has
num_nodes_snapshot = []
# test coordinates for each snapshot
test_coords_snapshots = []
# test values for each snapshot
test_values_snapshots = []
# high bandwidth test coordinates for each snapshot
test_thres_coords_snapshots = []
# high bandwidth test values for each snapshot
test_thres_values_snapshots = []

# define path for test data
test_path = snapshot_prefix + str(test_subgraph) + '.csv'
# define test snapshot
graph_ground_truth = nx.read_weighted_edgelist(test_path, nodetype=int, delimiter=',')

# == preprocessing ==
for i in range(start_subgraph, end_subgraph):
    train_path = snapshot_prefix + str(i) + '.csv'

    # read adj
    graph = nx.read_weighted_edgelist(train_path, nodetype=int, delimiter=',')
    clear_low_nodes(graph)
    adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
    adj.eliminate_zeros()

    # prepare adj tensor (dense)
    adj_train_with_diag = adj + sps.identity(adj.shape[0], dtype=np.float32)
    adj_tensor = tf.Variable(adj_train_with_diag.todense(), dtype=tf.float32)

    # prepare adj normalized tensor (sparse)
    adj_norm = calc_normalized(adj_train_with_diag)
    indices = np.mat([adj_norm.row, adj_norm.col]).transpose()
    adj_norm_tensor = tf.SparseTensor(indices, adj_norm.data, adj_norm.shape)

    # create feature matrix (identity matrix)
    features = sps.identity(adj_norm.shape[0], dtype=np.float32, format='coo')

    # prepare feature tensor (sparse)
    indices = np.mat([features.row, features.col]).transpose()
    features_tensor = tf.SparseTensor(indices, features.data, features.shape)

    # load testset
    adj_ground_truth = nx.adjacency_matrix(graph_ground_truth, nodelist=sorted(graph.nodes()))
    adj_ground_truth.eliminate_zeros()
    adj_ground_truth = adj_ground_truth.todense()
    adj_coords, adj_values, _ = sparse_to_tuple(adj)
    for coords in adj_coords:
        adj_ground_truth[coords[0], coords[1]] = 0
        adj_ground_truth[coords[1], coords[0]] = 0
    adj_ground_truth = sps.csr_matrix(adj_ground_truth)
    adj_ground_truth_thres = adj_ground_truth.copy()
    adj_ground_truth_thres[adj_ground_truth_thres < config_test_weight_threshold] = 0
    adj_ground_truth_thres.eliminate_zeros()
    test_coords, test_values, _ = sparse_to_tuple(adj_ground_truth)
    test_coords_thres, test_values_thres, _ = sparse_to_tuple(adj_ground_truth_thres)

    # append everything to lists
    adj_snapshots.append(adj_tensor)
    adj_norm_snapshots.append(adj_norm_tensor)
    features_snapshots.append(features_tensor)
    num_nodes_snapshot.append(adj.shape[0])
    test_coords_snapshots.append(test_coords)
    test_values_snapshots.append(test_values)
    test_thres_coords_snapshots.append(test_coords_thres)
    test_thres_values_snapshots.append(test_values_thres)


# == model definition ==

class FirstLayer(tf.keras.layers.Layer):
    def __init__(self, adj_norm, shared_w0):
        super(FirstLayer, self).__init__()
        self.adj_norm = adj_norm
        self.w = shared_w0

    def call(self, inputs, **kwargs):
        xw = tf.sparse.sparse_dense_matmul(inputs, self.w)
        axw = tf.sparse.sparse_dense_matmul(self.adj_norm, xw)
        relu = tf.nn.relu(axw)
        return relu


class SecondLayer(tf.keras.layers.Layer):

    def __init__(self, units, adj_norm):
        super(SecondLayer, self).__init__()
        self.units = units
        self.adj_norm = adj_norm
        self.training = True

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=tf.keras.initializers.glorot_uniform(),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        x = tf.matmul(inputs, self.w)
        x = tf.sparse.sparse_dense_matmul(self.adj_norm, x)
        return x


class Encoder(tf.keras.Model):
    def __init__(self, adj_norm, embedding_size, shared_w0):
        super(Encoder, self).__init__()
        self.first_layer = FirstLayer(adj_norm, shared_w0)
        self.mean_layer = SecondLayer(embedding_size, adj_norm)
        self.std_layer = SecondLayer(embedding_size, adj_norm)

    def call(self, input_features, **kwargs):
        intermediate = self.first_layer(input_features)
        means = self.mean_layer(intermediate)
        stds = self.std_layer(intermediate)
        z = means + (tf.random.normal(shape=means.shape) * tf.exp(stds))
        return z, means, stds


class ThirdLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(ThirdLayer, self).__init__()

    def call(self, inputs, **kwargs):
        matmul = tf.matmul(inputs, inputs, transpose_b=True)
        flat = tf.reshape(matmul, [-1])
        return flat


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.third_layer = ThirdLayer()

    def call(self, input_features, **kwargs):
        return self.third_layer(input_features)


class Autoencoder(tf.keras.Model):
    def __init__(self, adj_norm, embedding_size, shared_w0):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(adj_norm, embedding_size, shared_w0)
        self.decoder = Decoder()

    def call(self, input_features, **kwargs):
        z, means, stds = self.encoder(input_features)
        reconstructed = self.decoder(z)
        return reconstructed, means, stds


# == experiment ==
for experiment in l_depth_options:
    # setup experiment parameters 
    l_depth = experiment

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    glorot_initializer = tf.keras.initializers.glorot_uniform()

    autoencoders = []
    pos_weights = []
    norms = []
    labels = []

    # calculate DVGAE weights
    for i in np.arange(num_snapshots):
        adj = adj_snapshots[i]
        adj_sum = tf.reduce_sum(adj)
        pos_weights.append(float((adj.shape[0] * adj.shape[0]) - adj_sum) / adj_sum)
        norms.append(adj.shape[0] * adj.shape[0] / float(((adj.shape[0] * adj.shape[0]) - adj_sum) * 2))
        labels.append(tf.reshape(adj, [-1]))

    print("start training with size", experiment)
    # lists that keep track of results
    snapshot_history = defaultdict(list)
    kl_loss_history = defaultdict(list)
    reconstructed_loss_history = defaultdict(list)
    mean_mse = []
    mean_mse_thres = []
    var_mse = []
    var_mse_thres = []

    mean_mae = []
    mean_mae_thres = []
    var_mae = []
    var_mae_thres = []

    autoencoders = defaultdict(list)
    kl_losses = {}

    # iterate over the training window
    for i in range(num_snapshots):
        mse_acc = []
        thres_mse_acc = []
        mae_acc = []
        thres_mae_acc = []
        # repeat every experiment
        for rep in range(repeat):
            print('snapshot', start_subgraph + i)
            # prepare shared weights
            if i > 0:
                last_trained_ae = autoencoders[i - 1][rep]
                prev_w0 = last_trained_ae.encoder.first_layer.w
                num_new_nodes = num_nodes_snapshot[i] - num_nodes_snapshot[i - 1]
                if num_new_nodes > 0:
                    glorot_weights = tf.Variable(
                        initial_value=glorot_initializer(shape=(num_new_nodes, intermediate_size), dtype=tf.float32),
                        trainable=True)
                    w0 = tf.concat([prev_w0, glorot_weights], axis=0)
                else:
                    w0 = prev_w0
            else:
                w0 = tf.Variable(
                    initial_value=glorot_initializer(shape=(num_nodes_snapshot[0], intermediate_size),
                                                     dtype=tf.float32),
                    trainable=True)

            # create autoencoder
            autoenc = Autoencoder(adj_norm_snapshots[i], embedding_size, w0)
            autoencoders[i].append(autoenc)
            features = features_snapshots[i]
            norm = norms[i]
            pos_weight = pos_weights[i]
            label = labels[i]
            num_nodes = num_nodes_snapshot[i]

            # train autoencoder
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    # forward pass
                    reconstructed, means, stds = autoenc(features)
                    # compute train error
                    reconstruction_loss = norm * tf.reduce_mean(
                        tf.nn.weighted_cross_entropy_with_logits(logits=reconstructed, labels=label,
                                                                 pos_weight=pos_weight))
                    kl_self_loss = tf.abs((0.5 / num_nodes_snapshot[i]) * tf.reduce_mean(
                        tf.reduce_sum(1 + 2 * stds - tf.square(means) - tf.square(tf.exp(stds)), 1)))
                    kl_loss = 0
                    if i == 0:
                        kl_loss += kl_self_loss
                    else:
                        for l in range(i - 1, max(-1, i - 1 - l_depth), -1):
                            prev_kl = kl_losses[l]
                            kl_loss += (kl_self_loss + prev_kl) / 2
                    kl_losses[i] = kl_loss

                    step_loss = reconstruction_loss + kl_loss
                    snapshot_history[i].append(step_loss)
                    kl_loss_history[i].append(kl_loss)
                    reconstructed_loss_history[i].append(reconstruction_loss)

                # propagate gradients
                gradients = tape.gradient(step_loss, autoenc.trainable_variables)
                gradient_variables = zip(gradients, autoenc.trainable_variables)
                opt.apply_gradients(gradient_variables)

            # measure test error after training
            reconstructed, embeddings, stds = autoenc(features)
            mse_score = calc_mse(test_coords_snapshots[i], test_values_snapshots[i], embeddings).numpy().tolist()
            mse_score_thres = calc_mse(test_thres_coords_snapshots[i], test_thres_values_snapshots[i],
                                       embeddings).numpy().tolist()
            mae_score = calc_mae(test_coords_snapshots[i], test_values_snapshots[i], embeddings).numpy().tolist()
            mae_score_thres = calc_mae(test_thres_coords_snapshots[i], test_thres_values_snapshots[i],
                                       embeddings).numpy().tolist()
            mse_acc.append(mse_score)
            thres_mse_acc.append(mse_score_thres)
            mae_acc.append(mae_score)
            thres_mae_acc.append(mae_score_thres)

        mean_mse.append(mean(mse_acc))
        var_mse.append(pvariance(mse_acc))
        mean_mse_thres.append(mean(thres_mse_acc))
        var_mse_thres.append(pvariance(thres_mse_acc))

        mean_mae.append(mean(mae_acc))
        var_mae.append(pvariance(mae_acc))
        mean_mae_thres.append(mean(thres_mae_acc))
        var_mae_thres.append(pvariance(thres_mae_acc))

    # experiments are over, saving results into files
    save_path = output_dir + str(experiment) + '/'


    def save_data(list, filename):
        with open(save_path + filename, 'w') as handle:
            json.dump(list, handle)


    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_data(mean_mse, 'mean mse.json')
    save_data(var_mse, 'variance mse.json')
    save_data(mean_mse_thres, 'mean mse thres.json')
    save_data(var_mse_thres, 'variance mse thres.json')

    save_data(mean_mae, 'mean mae.json')
    save_data(var_mae, 'variance mae.json')
    save_data(mean_mae_thres, 'mean mae thres.json')
    save_data(var_mae_thres, 'variance mae thres.json')
