# import packages
import json
import random
import time
import warnings
from _collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

import networkx as nx
import numpy as np
import scipy.sparse as sps
import scipy.special as special
import scipy.stats as stats
import torch

config_device = 'cpu'
warnings.filterwarnings("ignore")

# first train snapshot
start_id = 19
# last train snapshot
stop_id = 30
# mu parameter
config_sqrt_pow = 4
# weight for high bandwidth edges
config_high_weight = 13
# input/output directory
snapshot_prefix = ''  # the input directory should point to the graph snapshots directory
output_dir = 'results/ABD/emb_size/'

# embedding size options
config_emb_size_options = [16, 32, 64, 128]
# window size
config_window_size = 2
# number of heads
config_nheads = 2

# threshold for high bandwidth test edges
config_test_weight_threshold = 0.4
# test snapshot
test_id = 299
# high bandwidth threshold (train)
config_label_goodness_threshold = 0.2
# learning rate
config_lr = 0.001
# weight for low bandwidth edges
config_low_weight = 0.1
# l2 norm multiplier
config_lambda = 1e-4
# number of train iterations
config_num_epochs = 100
# use boxcox transformation
config_use_boxcox = True
# use attention weights
config_use_att_weights = False
# dropout ratio
config_dropout = 0.1
# leakyRELU alpha
config_alpha = 0.1
# train with negative sampling
config_neg_sampling = False

config_repeat = 5
config_sample_nodes = 0
config_edges_per_node = 0

torch.set_num_threads(6)


# == model ==

class IDorder:
    def __init__(self):
        self.id_map = {}
        self.counter = 0

    def add(self, new_ids):
        for new_id in new_ids:
            if new_id not in self.id_map.keys():
                self.id_map[new_id] = self.counter
                self.counter += 1

    def get_map(self):
        return self.id_map


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(AttentionLayer, self).__init__()
        self.device = config_device
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = torch.nn.Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.W)
        self.a = torch.nn.parameter.Parameter(torch.zeros(size=(2 * out_features, 1)), requires_grad=True)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        if config_use_att_weights:
            attention = torch.mul(adj, e)
        else:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0., e, zero_vec)

        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        h_prime = torch.matmul(attention, h)

        result = torch.nn.functional.elu(h_prime) if self.concat else h_prime
        return result


class Attention_Model(torch.nn.Module):
    def __init__(self, cols, dropout, alpha, nheads):
        super().__init__()
        self.device = config_device
        self.dropout = dropout
        self.attention_blocks = []
        for head in range(nheads):
            block = AttentionLayer(cols, cols, dropout=self.dropout, alpha=alpha, concat=False)
            self.attention_blocks.append(block)
            self.add_module('block_{}'.format(head), block)
        self.output_block = AttentionLayer(nheads * cols, cols, dropout=self.dropout, alpha=alpha, concat=True)
        self.add_module('output_head', self.output_block)
        self.trainers = []
        for _ in range(config_window_size):
            trainer = Trainer()
            self.trainers.append(trainer)
        self.init_weights = None
        params = list(self.parameters())
        self.opt = torch.optim.Adam(params, lr=config_lr, weight_decay=config_lambda)

    def update(self, adj_l, coords_l, values_l, rows, cols):
        self.adj_list = [self.increase_size(adj, adj_l[-1].size()) for adj in adj_l]
        init_f = torch.nn.init.xavier_uniform_
        if self.init_weights is None:
            self.init_weights = torch.FloatTensor(rows, cols).to(device=config_device)
            init_f(self.init_weights)
        else:
            pad_size = rows - self.init_weights.size(0)
            pad = torch.FloatTensor(pad_size, cols).to(device=config_device)
            init_f(pad)
            self.init_weights = torch.cat([self.init_weights, pad], dim=0)

        for window in range(config_window_size):
            self.trainers[window].update(coords_l[window], values_l[window])

    def increase_size(self, matrix, size):
        z = torch.zeros(size=size)
        m_size = matrix.size()
        z[:m_size[0], :m_size[1]] = matrix
        return z

    def attention(self, weights, adj):
        intermediate = torch.cat([block(weights, adj) for block in self.attention_blocks], dim=1)
        emb = self.output_block(intermediate, adj)
        return emb

    def forward(self):
        embeddings = self.init_weights
        for window in range(config_window_size):
            adj = self.adj_list[window]
            embeddings = self.attention(embeddings, adj)
        return embeddings

    def calc_mse(self, coords, labels):
        embeddings = self.forward()
        return self.trainers[-1].calc_mse(coords, labels, embeddings)

    def calc_mae(self, coords, labels):
        embeddings = self.forward()
        return self.trainers[-1].calc_mae(coords, labels, embeddings)

    def window_train(self):
        for epoch in range(config_num_epochs):
            self.opt.zero_grad()
            embeddings = self.forward()
            loss = self.trainers[-1].evaluate(embeddings)
            loss.backward()
            self.opt.step()


class MyModel:
    def __init__(self):
        self.goodness_threshold = 0

    def update(self, goodness_threshold):
        self.goodness_threshold = goodness_threshold

    def predict(self, src_indices, dst_indices, embeddings):
        """Predicts the weight of given edges"""
        self.embeddings = embeddings
        src_emb = torch.index_select(self.embeddings, 0, src_indices)
        dst_emb = torch.index_select(self.embeddings, 0, dst_indices)
        product = src_emb * dst_emb
        predictions = torch.sum(product, dim=1)
        predictions = predictions.sigmoid()
        predictions = predictions.pow(1 / config_sqrt_pow)
        return predictions

    def evaluate(self, embeddings, center_nodes, neighbor_nodes, labels):
        """Returns the MSE between the predictions of given edges and the true labels"""
        # forward pass
        predictions = self.predict(center_nodes, neighbor_nodes, embeddings)

        # if config_neg_sampling:
        #     mse = torch.nn.MSELoss()
        #     pred_error = mse(predictions, labels)
        # else:
        low_weight = config_low_weight * torch.ones_like(labels)
        high_weight = config_high_weight * torch.ones_like(labels)
        # low_weight = torch.ones_like(labels)
        # high_weight = self.pos_weight * torch.ones_like(labels)
        weight = torch.where(labels > self.goodness_threshold, high_weight, low_weight)
        pred_error = torch.mean(weight * (predictions - labels).pow(2)).sqrt()
        return pred_error


class Trainer:
    # supports only incremental graphs
    def __init__(self):
        # these must be updated in each timestep
        self.graph = {}
        self.num_nodes = 0
        self.root_nodes = []
        self.goodness_threshold = 0

        self.model = MyModel()

    def update(self, edgelist, values):
        self.train_data = None
        new_values = values
        self.goodness_threshold = config_label_goodness_threshold
        if config_use_boxcox:
            adj_values_boxcox, opt_l = stats.boxcox(values)
            self.minval = np.min(adj_values_boxcox)
            self.diff = np.ptp(adj_values_boxcox)
            self.opt_l = opt_l
            adj_values_boxcox_norm = (adj_values_boxcox - np.min(adj_values_boxcox)) / np.ptp(adj_values_boxcox)
            new_values = adj_values_boxcox_norm

            dummy_list = [config_label_goodness_threshold, 1]
            dummy_list = np.array(dummy_list)
            dummy_list = stats.boxcox(dummy_list, self.opt_l)
            dummy_list = (dummy_list - self.minval) / self.diff
            self.goodness_threshold = dummy_list[0]

        self.read_edges(edgelist, new_values)
        self.model.update(self.goodness_threshold)

    def read_edges(self, edgelist, values):
        def add_to_graph(src, dst, val, graph):
            if src not in graph:
                graph[src] = {}
            if dst not in graph:
                graph[dst] = {}
            graph[src][dst] = val
            graph[dst][src] = val

        self.graph = {}
        for edge, value in zip(edgelist, values):
            add_to_graph(edge[0], edge[1], value, self.graph)
        self.root_nodes = list(self.graph.keys())
        self.num_nodes = len(self.root_nodes)

    def evaluate(self, embeddings):
        if config_neg_sampling:
            src, dst, vals = self.load_sample_edges()
        else:
            src, dst, vals = self.load_train_edges()
        src = torch.LongTensor(src).to(device=config_device)
        dst = torch.LongTensor(dst).to(device=config_device)
        vals = torch.FloatTensor(vals).to(device=config_device)

        pred_error = self.model.evaluate(embeddings, src, dst, vals)
        return pred_error

    def load_sample_edges(self):
        if self.train_data is None:
            low_src = []
            low_dst = []
            low_vals = []
            high_src = []
            high_dst = []
            high_vals = []

            for node in self.root_nodes:
                connections = list(self.graph[node].keys())
                values = list(self.graph[node].values())

                low_values = [val for val in values if val <= self.goodness_threshold]
                low_connections = [connections[counter] for counter, val in enumerate(values) if
                                   val <= self.goodness_threshold]
                high_values = [val for val in values if val > self.goodness_threshold]
                high_connections = [connections[counter] for counter, val in enumerate(values) if
                                    val > self.goodness_threshold]

                low_src.extend([node] * len(low_values))
                low_dst.extend(low_connections)
                low_vals.extend(low_values)

                high_src.extend([node] * len(high_values))
                high_dst.extend(high_connections)
                high_vals.extend(high_values)
            self.train_data = {}
            self.train_data['low_src'] = low_src
            self.train_data['low_dst'] = low_dst
            self.train_data['low_vals'] = low_vals
            self.train_data['high_src'] = high_src
            self.train_data['high_dst'] = high_dst
            self.train_data['high_vals'] = high_vals

        low_src = self.train_data['low_src']
        low_dst = self.train_data['low_dst']
        low_vals = self.train_data['low_vals']
        high_src = self.train_data['high_src']
        high_dst = self.train_data['high_dst']
        high_vals = self.train_data['high_vals']

        sample_size = len(high_vals)
        comb = list(zip(low_src, low_dst, low_vals))
        random.shuffle(comb)
        low_src, low_dst, low_vals = zip(*comb)
        low_src = list(low_src)
        low_dst = list(low_dst)
        low_vals = list(low_vals)
        low_src = low_src[:sample_size]
        low_dst = low_dst[:sample_size]
        low_vals = low_vals[:sample_size]
        src = low_src + high_src
        dst = low_dst + high_dst
        vals = low_vals + high_vals
        comb = list(zip(src, dst, vals))
        random.shuffle(comb)
        src, dst, vals = zip(*comb)
        return src, dst, vals

    def load_train_edges(self):
        if self.train_data is None:
            sample_nodes_size = config_sample_nodes if config_sample_nodes > 0 else self.num_nodes
            sample_edges_size = config_edges_per_node if config_edges_per_node > 0 else self.num_nodes

            if sample_nodes_size == self.num_nodes:
                sample_nodes = self.root_nodes
            else:
                sample_nodes = random.sample(self.root_nodes, sample_nodes_size)

            src = []
            dst = []
            vals = []
            for node in sample_nodes:
                connections = list(self.graph[node].keys())
                values = list(self.graph[node].values())

                if len(connections) > sample_edges_size:
                    comb = list(zip(connections, values))
                    random.shuffle(comb)
                    connections, values = zip(*comb)
                    connections = connections[:sample_edges_size]
                    values = values[:sample_edges_size]

                src.extend([node] * len(connections))
                dst.extend(connections)
                vals.extend(values)

            comb = list(zip(src, dst, vals))
            random.shuffle(comb)
            src, dst, vals = zip(*comb)
            self.train_data = {'src': src, 'dst': dst, 'vals': vals}
        else:
            src = self.train_data['src']
            dst = self.train_data['dst']
            vals = self.train_data['vals']
        return src, dst, vals

    def get_predictions_for(self, coords, embeddings):
        l1 = torch.from_numpy(coords[:, 0]).long().to(device=config_device)
        l2 = torch.from_numpy(coords[:, 1]).long().to(device=config_device)
        with torch.no_grad():
            predictions = self.model.predict(l1, l2, embeddings)
        return predictions

    def calc_mse(self, coords, labels, embeddings):
        predictions = self.get_predictions_for(coords, embeddings)
        predictions = predictions.clamp(min=0.0001, max=1).cpu()
        if config_use_boxcox:
            predictions = (predictions * self.diff) + self.minval
            predictions = special.inv_boxcox(predictions, self.opt_l)

        # pl.hist(labels, label='labels')
        # pl.hist(predictions, label='predictions')
        # pl.legend()
        # pl.show()

        labels = torch.from_numpy(labels)
        mse = torch.nn.MSELoss()
        score = mse(predictions, labels)
        return score

    def calc_mae(self, coords, labels, embeddings):
        predictions = self.get_predictions_for(coords, embeddings)
        predictions = predictions.clamp(min=0.0001, max=1).cpu()
        if config_use_boxcox:
            predictions = (predictions * self.diff) + self.minval
            predictions = special.inv_boxcox(predictions, self.opt_l)

        labels = torch.from_numpy(labels)
        mae = torch.mean(torch.abs(labels - predictions))
        return mae


def sparse_to_tuple(sparse_mx):
    sparse_mx = sps.triu(sparse_mx)
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


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


if __name__ == '__main__':
    # create id map
    id_map = IDorder()
    # define test snapshot
    test_path = snapshot_prefix + str(test_id) + '.csv'
    # read test snapshot
    graph_ground_truth = nx.read_weighted_edgelist(test_path, nodetype=int, delimiter=',')

    adj_list = []
    dim_list = []
    train_coords_list = []
    train_values_list = []
    test_coords_list = []
    test_values_list = []
    thres_test_coords_list = []
    thres_test_values_list = []

    for i in range(start_id, stop_id):
        train_path = snapshot_prefix + str(i) + '.csv'
        # read train adj
        graph = nx.read_weighted_edgelist(train_path, nodetype=int, delimiter=',')
        clear_low_nodes(graph)
        test_graph = graph_ground_truth.subgraph(graph.nodes()).copy()
        id_map.add((list(graph.nodes())))
        graph = nx.relabel_nodes(graph, id_map.get_map(), copy=True)
        test_graph = nx.relabel_nodes(test_graph, id_map.get_map(), copy=True)

        adj = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
        adj.eliminate_zeros()
        adj_tensor = torch.tensor(adj.todense(), dtype=torch.float).to(device=config_device)

        dim = adj.shape[0]

        adj_ground_truth = nx.adjacency_matrix(test_graph, nodelist=sorted(test_graph.nodes()))
        adj_ground_truth.eliminate_zeros()

        adj_coords, adj_values, _ = sparse_to_tuple(adj)
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

        adj_list.append(adj_tensor)
        dim_list.append(dim)
        train_coords_list.append(adj_coords)
        train_values_list.append(adj_values)
        test_coords_list.append(test_coords)
        test_values_list.append(test_values)
        thres_test_coords_list.append(test_coords_thres)
        thres_test_values_list.append(test_values_thres)

    for experiment in config_emb_size_options:
        config_emb_size = experiment
        print("start training with window size", experiment)

        mse_scores = defaultdict(list)
        mse_thres_scores = defaultdict(list)
        mae_scores = defaultdict(list)
        mae_thres_scores = defaultdict(list)
        exec_times = defaultdict(list)
        for rep in range(config_repeat):
            att_model = Attention_Model(config_emb_size, config_dropout, config_alpha, config_nheads).to(
                device=config_device)
            for i in range(stop_id - start_id - config_window_size + 1):
                att_model.train()
                window_stop = i + config_window_size
                window_adj = adj_list[i: window_stop]
                window_dim = dim_list[i: window_stop]
                window_train_coords = train_coords_list[i: window_stop]
                window_train_values = train_values_list[i: window_stop]
                window_test_coords = test_coords_list[i: window_stop]
                window_test_values = test_values_list[i: window_stop]
                window_thres_test_coords = thres_test_coords_list[i: window_stop]
                window_thres_test_values = thres_test_values_list[i: window_stop]

                start = time.time()
                att_model.update(window_adj, window_train_coords, window_train_values, window_dim[-1], config_emb_size)
                att_model.window_train()
                end = time.time()
                exec = end - start
                exec_times[i].append(exec)

                att_model.eval()
                with torch.no_grad():
                    mse_score = att_model.calc_mse(window_test_coords[-1], window_test_values[-1])
                    mse_scores[i].append(mse_score.data.numpy().tolist())
                    mae_score = att_model.calc_mae(window_test_coords[-1], window_test_values[-1])
                    mae_scores[i].append(mae_score.data.numpy().tolist())
                    mse_score_thres = att_model.calc_mse(window_thres_test_coords[-1], window_thres_test_values[-1])
                    mse_thres_scores[i].append(mse_score_thres.data.numpy().tolist())
                    mae_score_thres = att_model.calc_mae(window_thres_test_coords[-1], window_thres_test_values[-1])
                    mae_thres_scores[i].append(mae_score_thres.data.numpy().tolist())
                    print('epoch', start_id + i + config_window_size, 'total mse', mse_score.data.numpy(), 'thres mse',
                          mse_score_thres.data.numpy(), 'exec time', exec)

        mean_mse = []
        mean_mse_thres = []
        var_mse = []
        var_mse_thres = []
        mean_exec_time = []
        var_exec_time = []

        mean_mae = []
        mean_mae_thres = []
        var_mae = []
        var_mae_thres = []
        for i in range(stop_id - start_id - config_window_size + 1):
            it_mse = mse_scores[i]
            it_mean_mse = mean(it_mse)
            it_var_mse = pvariance(it_mse)
            mean_mse.append(it_mean_mse)
            var_mse.append(it_var_mse)

            it_mse_thres = mse_thres_scores[i]
            it_mean_mse_thres = mean(it_mse_thres)
            it_var_mse_thres = pvariance(it_mse_thres)
            mean_mse_thres.append(it_mean_mse_thres)
            var_mse_thres.append(it_var_mse_thres)

            it_mae = mae_scores[i]
            it_mean_mae = mean(it_mae)
            it_var_mae = pvariance(it_mae)
            mean_mae.append(it_mean_mae)
            var_mae.append(it_var_mae)

            it_mae_thres = mae_thres_scores[i]
            it_mean_mae_thres = mean(it_mae_thres)
            it_var_mae_thres = pvariance(it_mae_thres)
            mean_mae_thres.append(it_mean_mae_thres)
            var_mae_thres.append(it_var_mae_thres)

            it_exec_times = exec_times[i]
            it_mean_exec = mean(it_exec_times)
            it_var_exec = pvariance(it_exec_times)
            mean_exec_time.append(it_mean_exec)
            var_exec_time.append(it_var_exec)

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
