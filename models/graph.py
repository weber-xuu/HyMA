from collections import defaultdict, deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, HypergraphConv
from transformers import AutoTokenizer
from transformers.activations import ACT2FN

import utils

from .hyperbolic_head import _parse_curvature_init
from .lorentz_ops import exp_map0, log_map0, smooth_clip_tangent_norm


def _normalize_graph_space(graph_space):
    normalized = (graph_space or 'euclidean').strip().lower()
    mapping = {
        'euclidean': 'euclidean',
        'euc': 'euclidean',
        'tangent_hyperbolic': 'tangent_hyperbolic',
        'tangent': 'tangent_hyperbolic',
        'hyperbolic_tangent': 'tangent_hyperbolic',
    }
    if normalized not in mapping:
        supported = ', '.join(['euclidean', 'tangent_hyperbolic'])
        raise ValueError(f'Unsupported graph space: {graph_space}. Supported values: {supported}.')
    return mapping[normalized]


class GraphAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        output_attentions: bool = False,
        extra_attn=None,
        only_attn=False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights = attn_weights + extra_attn

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, graph_type):
        super().__init__()
        self.config = config
        self.graph_type = graph_type

        if self.graph_type == 'graphormer':
            self.graph = GraphAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
        elif self.graph_type == 'GCN':
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            self.graph = GATConv(config.hidden_size, config.hidden_size, heads=1)
        elif self.graph_type == 'hypergraph':
            self.graph = HypergraphConv(config.hidden_size, config.hidden_size)
        else:
            raise NotImplementedError(f'Unsupported graph layer type: {self.graph_type}')

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'graphormer':
            label_emb, _, _ = self.graph(
                hidden_states=label_emb,
                attention_mask=None,
                output_attentions=False,
                extra_attn=extra_attn,
            )
            label_emb = F.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = self.layer_norm(residual + label_emb)

            residual = label_emb
            label_emb = self.activation_fn(self.fc1(label_emb))
            label_emb = F.dropout(label_emb, p=self.activation_dropout, training=self.training)
            label_emb = self.fc2(label_emb)
            label_emb = F.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = self.final_layer_norm(residual + label_emb)
        elif self.graph_type in ('GCN', 'GAT'):
            label_states = self.graph(label_emb.squeeze(0), edge_index=extra_attn)
            label_states = F.dropout(label_states, p=self.dropout, training=self.training)
            label_emb = self.layer_norm(residual + label_states.unsqueeze(0))
        elif self.graph_type == 'hypergraph':
            label_states = self.graph(label_emb.squeeze(0), hyperedge_index=extra_attn)
            label_states = F.dropout(label_states, p=self.dropout, training=self.training)
            label_emb = self.layer_norm(residual + label_states.unsqueeze(0))
        else:
            raise NotImplementedError(f'Unsupported graph layer type: {self.graph_type}')

        return label_emb


class GraphEncoder(nn.Module):
    def __init__(
        self,
        config,
        graph_type='GAT',
        layer=1,
        path_list=None,
        data_path=None,
        graph_space='euclidean',
        node_depth_ids=None,
        num_depths=1,
        curvature_init=1.0,
        curvature_init_by_depth=None,
        per_depth_curvature=False,
        tangent_clip=2.0,
    ):
        super().__init__()
        self.config = config
        self.graph_type = utils.normalize_graph_type(graph_type)
        self.graph_space = _normalize_graph_space(graph_space)
        self.hir_layers = nn.ModuleList([GraphLayer(config, self.graph_type) for _ in range(layer)])
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
        self.num_depths = max(int(num_depths), 1)
        self.per_depth_curvature = per_depth_curvature
        self.tangent_clip = tangent_clip

        flat_nodes = [node for edge in (path_list or []) for node in edge if node >= 0]
        self.node_num = max(flat_nodes) + 1 if flat_nodes else self.num_labels
        self.path_list = nn.Parameter(self._build_edge_index(path_list), requires_grad=False)
        if node_depth_ids is None:
            node_depth_ids = torch.zeros(self.node_num, dtype=torch.long)
        self.node_depth_ids = nn.Parameter(torch.as_tensor(node_depth_ids, dtype=torch.long), requires_grad=False)

        if self.graph_space == 'tangent_hyperbolic':
            init_values = _parse_curvature_init(curvature_init, curvature_init_by_depth, self.num_depths)
            raw_init = [math.log(math.expm1(value)) for value in init_values]
            if self.per_depth_curvature:
                raw_tensor = torch.tensor(raw_init, dtype=torch.float32)
            else:
                raw_tensor = torch.tensor([raw_init[0]], dtype=torch.float32)
            self.raw_curvature_by_depth = nn.Parameter(raw_tensor)

        if self.graph_type == 'graphormer':
            self._init_graphormer_buffers(data_path)
        elif self.graph_type == 'hypergraph':
            self._init_hypergraph_index()

    def get_all_curvatures(self):
        if self.graph_space != 'tangent_hyperbolic':
            return None
        base = F.softplus(self.raw_curvature_by_depth) + 1e-4
        if self.per_depth_curvature:
            return base
        return base.expand(self.num_depths)

    def get_node_curvatures(self, device=None, dtype=None):
        all_curvatures = self.get_all_curvatures()
        if all_curvatures is None:
            return None
        node_curvatures = all_curvatures.index_select(0, self.node_depth_ids)
        if device is not None:
            node_curvatures = node_curvatures.to(device=device)
        if dtype is not None:
            node_curvatures = node_curvatures.to(dtype=dtype)
        return node_curvatures

    def _clip_tangent_norm(self, vectors):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return vectors
        return smooth_clip_tangent_norm(vectors, max_norm=self.tangent_clip)

    def _prepare_tangent_hidden(self, hidden_states):
        if self.graph_space != 'tangent_hyperbolic':
            return hidden_states

        squeeze_batch = hidden_states.dim() == 3
        node_states = hidden_states.squeeze(0) if squeeze_batch else hidden_states
        node_states = self._clip_tangent_norm(node_states)
        node_curvatures = self.get_node_curvatures(device=node_states.device, dtype=node_states.dtype).unsqueeze(-1)
        node_states = log_map0(exp_map0(node_states, curv=node_curvatures), curv=node_curvatures)
        return node_states.unsqueeze(0) if squeeze_batch else node_states

    def _build_edge_index(self, path_list):
        valid_edges = []
        for parent, child in path_list or []:
            if parent == -1:
                continue
            if 0 <= parent < self.node_num and 0 <= child < self.node_num:
                valid_edges.append((parent, child))
        if not valid_edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(valid_edges, dtype=torch.long).transpose(0, 1).contiguous()

    def _build_node_texts(self, data_path):
        label_dict = utils.load_label_dict(data_path)
        node_texts = []
        for node_id in range(self.node_num):
            if node_id < self.num_labels:
                node_texts.append(str(label_dict[node_id]))
            else:
                node_texts.append(f'depth_{node_id - self.num_labels}')
        encoded = self.tokenizer(
            node_texts,
            padding='longest',
            truncation=True,
            max_length=32,
            return_tensors='pt',
        )
        self.node_name_ids = nn.Parameter(encoded['input_ids'], requires_grad=False)
        self.node_name_mask = nn.Parameter(encoded['attention_mask'], requires_grad=False)

    def _build_shortest_paths(self):
        adjacency = defaultdict(list)
        for parent, child in self.path_list.transpose(0, 1).tolist():
            adjacency[parent].append(child)
            adjacency[child].append(parent)

        routes = {}
        distances = torch.zeros((self.node_num, self.node_num), dtype=torch.long)
        max_route_len = 1

        for source in range(self.node_num):
            parents = {source: -1}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor in parents:
                        continue
                    parents[neighbor] = current
                    queue.append(neighbor)

            for target in range(self.node_num):
                if target not in parents:
                    route = [source] if source == target else []
                else:
                    route = []
                    cursor = target
                    while cursor != -1:
                        route.append(cursor)
                        if cursor == source:
                            break
                        cursor = parents[cursor]
                    route.reverse()

                if not route:
                    route = [source]

                routes[(source, target)] = route
                distances[source, target] = max(len(route) - 1, 0)
                max_route_len = max(max_route_len, len(route))

        return distances, routes, max_route_len

    def _init_graphormer_buffers(self, data_path):
        self._build_node_texts(data_path)
        distances, routes, max_route_len = self._build_shortest_paths()
        pad_idx = self.node_num

        edge_mat = torch.full((self.node_num, self.node_num, max_route_len), pad_idx, dtype=torch.long)
        path_lengths = torch.ones((self.node_num, self.node_num), dtype=torch.long)
        for source in range(self.node_num):
            for target in range(self.node_num):
                route = routes[(source, target)]
                edge_mat[source, target, :len(route)] = torch.tensor(route, dtype=torch.long)
                path_lengths[source, target] = max(len(route), 1)

        self.id_embedding = nn.Embedding(self.node_num, self.config.hidden_size, 0)
        self.distance_embedding = nn.Embedding(int(distances.max().item()) + 1, 1, 0)
        self.edge_embedding = nn.Embedding(self.node_num + 1, 1, padding_idx=pad_idx)
        self.node_ids = nn.Parameter(torch.arange(self.node_num, dtype=torch.long), requires_grad=False)
        self.edge_mat = nn.Parameter(edge_mat.view(-1, max_route_len), requires_grad=False)
        self.distance_mat = nn.Parameter(distances.view(-1), requires_grad=False)
        self.path_lengths = nn.Parameter(path_lengths.view(-1), requires_grad=False)

    def _init_hypergraph_index(self):
        parent_to_children = defaultdict(list)
        for parent, child in self.path_list.transpose(0, 1).tolist():
            if 0 <= parent < self.node_num and 0 <= child < self.node_num:
                parent_to_children[parent].append(child)

        node_idx_list = []
        hyperedge_idx_list = []
        hyperedge_id_counter = 0
        for parent, children in parent_to_children.items():
            if not children:
                continue
            for node_id in [parent] + children:
                node_idx_list.append(node_id)
                hyperedge_idx_list.append(hyperedge_id_counter)
            hyperedge_id_counter += 1

        if node_idx_list:
            hyperedge_index = torch.tensor([node_idx_list, hyperedge_idx_list], dtype=torch.long)
        else:
            hyperedge_index = torch.empty((2, 0), dtype=torch.long)
        self.hyperedge_index = nn.Parameter(hyperedge_index, requires_grad=False)

    def forward(self, label_emb, embeddings):
        if self.graph_type == 'graphormer':
            label_name_emb = embeddings(self.node_name_ids)
            label_mask = self.node_name_mask.unsqueeze(-1)
            label_emb = label_emb + (label_name_emb * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1)
            label_emb = label_emb + self.id_embedding(self.node_ids)

            distance_bias = self.distance_embedding(self.distance_mat)
            edge_bias = self.edge_embedding(self.edge_mat).sum(dim=1) / self.path_lengths.unsqueeze(-1).clamp_min(1)
            extra_attn = (distance_bias + edge_bias).view(self.node_num, self.node_num)
        elif self.graph_type in ('GCN', 'GAT'):
            extra_attn = self.path_list
        elif self.graph_type == 'hypergraph':
            extra_attn = self.hyperedge_index
        else:
            raise NotImplementedError(f'Unsupported graph encoder type: {self.graph_type}')

        hidden_states = self._prepare_tangent_hidden(label_emb).unsqueeze(0)
        for hir_layer in self.hir_layers:
            hidden_states = hir_layer(hidden_states, extra_attn)
            hidden_states = self._prepare_tangent_hidden(hidden_states)
        return hidden_states.squeeze(0)
