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
from statistics import mean, pvariance

# input/output directory
snapshot_prefix = ''  # the input directory should point to the graph snapshots directory
output_dir = 'results/EGCN/performance/'

# == parameters ==
# threshold for high bandwidth test edges
config_test_weight_threshold = 0.4
# first train snapshot
start_subgraph = 9
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
# training window size
window_size = 2
# number of times the experiment is repeated
repeat = 5


# creates the degree matrix of the input
def degree_matrix(matrix, max_degree):
    rowsum = np.count_nonzero(matrix, axis=1)
    features = np.zeros((matrix.shape[0], max_degree), dtype=np.float32)
    for i in range(rowsum.shape[0]):
        features[i][rowsum[i] - 1] = 1
    return features


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


def read_adjacency(file):
    graph = nx.read_weighted_edgelist(file, delimiter=',', nodetype=int)
    node_list = list(graph.nodes()).sort()
    adj = nx.adjacency_matrix(graph, nodelist=node_list)
    return adj


# calculates the mean square error
def calc_mse(coords, values, reconstructed):
    reconstructed = tf.sigmoid(reconstructed)
    predictions = []
    for src, dst in coords:
        pred = reconstructed[src][dst].numpy().tolist()
        predictions.append(pred)

    predictions = tf.Variable(predictions)
    mse = tf.keras.losses.MSE(values, predictions)
    return mse


# calculates the mean absolute error
def calc_mae(coords, values, reconstructed):
    reconstructed = tf.sigmoid(reconstructed)
    predictions = []
    for src, dst in coords:
        pred = reconstructed[src][dst].numpy().tolist()
        predictions.append(pred)

    predictions = tf.Variable(predictions)
    mae = tf.reduce_mean(tf.abs(values - predictions))
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
    adj_coords, adj_values, adj_shape = sparse_to_tuple(adj)

    adj_train_with_diag = adj + sps.identity(adj.shape[0], dtype=np.float32)
    # prepare adj normalized tensor (sparse)
    adj_norm = calc_normalized(adj_train_with_diag)
    adj_norm_tensor = tf.convert_to_tensor(adj_norm.todense(), dtype=tf.float32)

    # load testset
    adj_ground_truth = nx.adjacency_matrix(graph_ground_truth, nodelist=sorted(graph.nodes()))
    adj_ground_truth.eliminate_zeros()
    adj_ground_truth = adj_ground_truth.todense()
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
    adj_snapshots.append(adj.todense().astype(np.float32))
    adj_norm_snapshots.append(adj_norm_tensor)
    num_nodes_snapshot.append(adj.shape[0])
    test_coords_snapshots.append(test_coords)
    test_values_snapshots.append(test_values)
    test_thres_coords_snapshots.append(test_coords_thres)
    test_thres_values_snapshots.append(test_values_thres)


# == model definition ==

class EGCN_O(tf.keras.Model):
    def __init__(self, layer0_feats, layer1_feats, layer2_feats):
        super(EGCN_O, self).__init__()
        self.GRCU1 = GRCU_O(layer0_feats, layer1_feats, tf.nn.relu)
        self.GRCU2 = GRCU_O(layer1_feats, layer2_feats, tf.identity)

    def call(self, adj_list, feats_list):
        intermediate = self.GRCU1(adj_list, feats_list)
        embeddings = self.GRCU2(adj_list, intermediate)
        return embeddings[-1]


class EGCN_H(tf.keras.Model):
    def __init__(self, feats_per_node, l1_feats, l2_feats):
        super(EGCN_H, self).__init__()
        self.GRCU1 = GRCU_H(feats_per_node, l1_feats, tf.nn.relu)
        self.GRCU2 = GRCU_H(l1_feats, l2_feats, tf.identity)

    def call(self, adj_list, nodes_list):
        intermediate = self.GRCU1(adj_list, nodes_list)
        embeddings = self.GRCU2(adj_list, intermediate)
        return embeddings[-1]


class GRCU_O(tf.keras.layers.Layer):
    def __init__(self, rows, cols, activation):
        super(GRCU_O, self).__init__()
        self.rows = rows
        self.cols = cols
        self.evolve_weights = Cell_O(rows, cols)
        self.activation = activation
        c_stdv = 1. / tf.sqrt(tf.cast(cols, dtype=tf.float32))
        self.GCN_init_weights = tf.random.uniform((rows, cols), minval=-c_stdv, maxval=c_stdv)

    def call(self, adj_list, feat_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, adj in enumerate(adj_list):
            feats = feat_list[t]
            # evolve weights
            GCN_weights = self.evolve_weights(GCN_weights)
            # GCN
            hw = tf.matmul(feats, GCN_weights)
            ahw = tf.matmul(adj, hw)
            emb = self.activation(ahw)
            out_seq.append(emb)
        return out_seq


class GRCU_H(tf.keras.layers.Layer):
    def __init__(self, rows, cols, activation):
        super(GRCU_H, self).__init__()
        self.rows = rows
        self.cols = cols
        self.evolve_weights = Cell_H(rows, cols)
        self.activation = activation
        c_stdv = 1. / tf.sqrt(tf.cast(cols, dtype=tf.float32))
        self.GCN_init_weights = tf.random.uniform((rows, cols), minval=-c_stdv, maxval=c_stdv)

    def call(self, adj_list, feat_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, adj in enumerate(adj_list):
            feats = feat_list[t]
            # evolve weights
            GCN_weights = self.evolve_weights(GCN_weights, feats)
            # GCN
            hw = tf.matmul(feats, GCN_weights)
            ahw = tf.matmul(adj, hw)
            emb = self.activation(ahw)
            out_seq.append(emb)
        return out_seq


class Cell_O(tf.keras.layers.Layer):
    def __init__(self, rows, cols):
        super(Cell_O, self).__init__()
        self.rows = rows
        self.cols = cols
        self.update = Gate(rows, cols, tf.nn.sigmoid)
        self.reset = Gate(rows, cols, tf.nn.sigmoid)
        self.htilda = Gate(rows, cols, tf.nn.tanh)

    def call(self, prev_Q):
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = ((1 - update) * prev_Q) + (update * h_cap)
        return new_Q


class Cell_H(tf.keras.layers.Layer):
    def __init__(self, rows, cols):
        super(Cell_H, self).__init__()
        self.rows = rows
        self.cols = cols
        self.update = Gate(rows, cols, tf.nn.sigmoid)
        self.reset = Gate(rows, cols, tf.nn.sigmoid)
        self.htilda = Gate(rows, cols, tf.nn.tanh)
        self.choose_topk = TopK()

    def call(self, prev_Q, prev_Z):
        k = prev_Q.shape[1]
        z_topk = self.choose_topk(prev_Z, k)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = ((1 - update) * prev_Q) + (update * h_cap)
        return new_Q


class Gate(tf.keras.layers.Layer):
    def __init__(self, rows, cols, activation):
        super(Gate, self).__init__()
        self.rows = rows
        self.cols = cols
        self.activation = activation

        r_stdv = 1. / tf.sqrt(tf.cast(self.rows, dtype=tf.float32))
        r_init = tf.random_uniform_initializer(minval=-r_stdv, maxval=r_stdv)

        self.W = self.add_weight(shape=(rows, rows), initializer=r_init, trainable=True)
        self.U = self.add_weight(shape=(rows, rows), initializer=r_init, trainable=True)
        self.B = self.add_weight(shape=(rows, cols), initializer=tf.initializers.zeros, trainable=True)

    def call(self, input, hidden):
        w_input = tf.matmul(self.W, input)
        u_hidden = tf.matmul(self.U, hidden)
        sum = w_input + u_hidden + self.B
        return self.activation(sum)


class TopK(tf.keras.layers.Layer):
    def __init__(self):
        super(TopK, self).__init__()

    def call(self, emb, k):
        cols = emb.shape[1]
        c_stdv = 1. / tf.sqrt(tf.cast(cols, dtype=tf.float32))
        p = tf.random.uniform((cols, 1), minval=-c_stdv, maxval=c_stdv)

        scores = tf.matmul(emb, p) / tf.norm(p)

        scores = tf.reshape(scores, shape=[-1])
        kmin = min(k, scores.shape[0])
        vals, topk_indices = tf.math.top_k(scores, kmin)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.shape[0] < k:
            diff = k - topk_indices.shape[0]
            rest = tf.ones((diff,), dtype=tf.float32) * tf.cast(topk_indices[-1], tf.float32)
            rest = tf.cast(rest, tf.int32)
            topk_indices = tf.concat([topk_indices, rest], axis=0)

        out = tf.gather(emb, topk_indices) * tf.nn.tanh(tf.reshape(tf.gather(scores, topk_indices), shape=(-1, 1)))
        return tf.transpose(out)


class Classifier(tf.keras.Model):
    def __init__(self, intermediate_size, original_size):
        super(Classifier, self).__init__()
        self.intermediate_layer = tf.keras.layers.Dense(units=intermediate_size,
                                                        activation=tf.nn.relu,
                                                        kernel_initializer=tf.random_uniform_initializer())
        self.reconstruction_layer = tf.keras.layers.Dense(units=original_size,
                                                          kernel_initializer=tf.random_uniform_initializer())

    def call(self, embeddings):
        intermediate = self.intermediate_layer(embeddings)
        reconstructed = self.reconstruction_layer(intermediate)
        return reconstructed


opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
mean_mse = []
mean_mse_thres = []
var_mse = []
var_mse_thres = []

mean_mae = []
mean_mae_thres = []
var_mae = []
var_mae_thres = []

# iterate over the training window
for i in range(num_snapshots - window_size + 1):
    print(start_subgraph + i)
    mse_acc = []
    thres_mse_acc = []
    mae_acc = []
    thres_mae_acc = []
    # repeat every experiment
    for rep in range(repeat):

        # construct features list
        features_list = []
        last_id = i + window_size - 1
        last_adj = adj_snapshots[last_id]
        max_degree = last_adj.shape[0] - 1
        for j in range(i, i + window_size):
            features = degree_matrix(adj_snapshots[j], max_degree)
            features_list.append(features)

        # adj norm list
        adj_norm_list = adj_norm_snapshots[i: i + window_size]

        # define model
        model = EGCN_H(max_degree, intermediate_size, embedding_size)

        # define classifier
        classifier = Classifier(intermediate_size, max_degree + 1)

        # train
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # run model
                embeddings = model(adj_norm_list, features_list)
                # produce predictions
                predictions = classifier(embeddings)
                # cross entropy
                labels = tf.reshape(last_adj, [-1])
                logits = tf.reshape(predictions, [-1])
                ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            # gradients
            gradients = tape.gradient(ce, model.trainable_variables + classifier.trainable_variables)
            gradient_variables = zip(gradients, model.trainable_variables + classifier.trainable_variables)
            opt.apply_gradients(gradient_variables)

        mse_score = calc_mse(test_coords_snapshots[last_id], test_values_snapshots[last_id],
                             predictions).numpy().tolist()
        mse_score_thres = calc_mse(test_thres_coords_snapshots[last_id], test_thres_values_snapshots[last_id],
                                   predictions).numpy().tolist()
        mae_score = calc_mae(test_coords_snapshots[last_id], test_values_snapshots[last_id],
                             predictions).numpy().tolist()
        mae_score_thres = calc_mae(test_thres_coords_snapshots[last_id], test_thres_values_snapshots[last_id],
                                   predictions).numpy().tolist()
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

save_path = output_dir + '/'


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
