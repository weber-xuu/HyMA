import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

import utils

from .graph import GraphEncoder
from .hyperbolic_head import HyperbolicClassifierHead
from .loss import multilabel_categorical_crossentropy


def _get_dataset_name(data_path):
    return os.path.basename(os.path.normpath(data_path))


def _load_id2label_map(data_path, dataset_name=None):
    dataset_name = dataset_name or _get_dataset_name(data_path)

    if dataset_name == 'nyt' and os.path.exists(os.path.join(data_path, 'label_dict.pt')):
        raw_label_dict = utils.torch_load_compat(os.path.join(data_path, 'label_dict.pt'))
        return {int(i): v for i, v in raw_label_dict.items()}

    raw_label_dict = utils.torch_load_compat(os.path.join(data_path, 'value_dict.pt'))
    if len(raw_label_dict) == 0:
        return {}

    sample_key = next(iter(raw_label_dict.keys()))
    if isinstance(sample_key, str):
        return {int(v): str(k) for k, v in raw_label_dict.items()}
    return {int(i): v for i, v in raw_label_dict.items()}


def _normalize_graph_type(graph_type):
    normalized = (graph_type or '').strip().lower()
    mapping = {
        '': '',
        'none': '',
        'gat': 'GAT',
        'gcn': 'GCN',
        'graphormer': 'graphormer',
        'hypergraph': 'hypergraph',
    }
    if normalized not in mapping:
        supported = ', '.join(['GAT', 'GCN', 'graphormer', 'hypergraph'])
        raise ValueError(f'Unsupported graph type: {graph_type}. Supported values: {supported}.')
    return mapping[normalized]


class GraphEmbedding(nn.Module):
    def __init__(
        self,
        config,
        embedding,
        new_embedding,
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
        self.graph_type = _normalize_graph_type(graph_type)
        self.graph_space = graph_space
        self.num_class = config.num_labels
        self.padding_idx = config.pad_token_id
        self.original_embedding = embedding

        if self.graph_type:
            self.graph = GraphEncoder(
                config,
                self.graph_type,
                layer,
                path_list=path_list,
                data_path=data_path,
                graph_space=graph_space,
                node_depth_ids=node_depth_ids,
                num_depths=num_depths,
                curvature_init=curvature_init,
                curvature_init_by_depth=curvature_init_by_depth,
                per_depth_curvature=per_depth_curvature,
                tangent_clip=tangent_clip,
            )

        new_embedding = torch.cat(
            [
                torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
                new_embedding,
            ],
            dim=0,
        )
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, freeze=False, padding_idx=0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = self.new_embedding.num_embeddings - 2 - self.num_class

    @property
    def weight(self):
        def build_weight():
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type:
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat([edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return build_weight

    @property
    def raw_weight(self):
        def build_weight():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return build_weight

    def forward(self, x):
        return F.embedding(x, self.weight(), self.padding_idx)


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class Prompt(nn.Module):
    def __init__(
        self,
        config,
        backbone_model,
        pretrained_model_name_or_path,
        graph_type='GAT',
        layer=1,
        path_list=None,
        data_path=None,
        depth2label=None,
        seloss_wight=None,
        use_label_description=False,
        orth_method='gram_schmidt',
        mlm_mask_strategy='legacy',
        classifier_head='euclidean',
        hyperbolic_dim=256,
        hyperbolic_alpha=0.3,
        depth_aware_hyper_alpha=False,
        hyperbolic_curvature_init=1.0,
        curvature_init_by_depth=None,
        per_depth_curvature=False,
        hyperbolic_logit_scale_init=1.0,
        hyperbolic_radius_clip=2.0,
        graph_space='euclidean',
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.config.name_or_path = pretrained_model_name_or_path
        self.name_or_path = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.bert = self._extract_backbone(backbone_model)
        self.cls = self._extract_lm_head(backbone_model)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(config.hidden_size)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)

        self.data_path = data_path
        self.dataset_name = _get_dataset_name(self.data_path)
        self.graph_type = _normalize_graph_type(graph_type)
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list or []
        self.depth2label = depth2label or {}
        self.num_depths = max(len(self.depth2label), 1)
        self.layer = layer
        self.use_label_description = use_label_description
        self.seloss_wight = seloss_wight
        self.orth_method = self._normalize_orth_method(orth_method)
        self.mlm_mask_strategy = self._normalize_mlm_mask_strategy(mlm_mask_strategy)
        self.classifier_head = self._normalize_classifier_head(classifier_head)
        self.hyperbolic_dim = hyperbolic_dim
        self.hyperbolic_alpha = hyperbolic_alpha
        self.depth_aware_hyper_alpha = depth_aware_hyper_alpha
        self.hyperbolic_curvature_init = hyperbolic_curvature_init
        self.curvature_init_by_depth = curvature_init_by_depth
        self.per_depth_curvature = per_depth_curvature
        self.hyperbolic_logit_scale_init = hyperbolic_logit_scale_init
        self.hyperbolic_radius_clip = hyperbolic_radius_clip
        self.graph_space = graph_space
        label_depth_ids = torch.zeros(self.num_labels, dtype=torch.long)
        for depth_idx, label_ids in self.depth2label.items():
            if not label_ids:
                continue
            label_depth_ids[torch.tensor(label_ids, dtype=torch.long)] = int(depth_idx)
        self.register_buffer('label_depth_ids', label_depth_ids, persistent=False)
        if self.classifier_head in ('hyperbolic', 'hybrid'):
            self.hyperbolic_head = HyperbolicClassifierHead(
                input_dim=config.hidden_size,
                hyperbolic_dim=hyperbolic_dim,
                num_depths=self.num_depths,
                curvature_init=hyperbolic_curvature_init,
                curvature_init_by_depth=curvature_init_by_depth,
                logit_scale_init=hyperbolic_logit_scale_init,
                tangent_clip=hyperbolic_radius_clip,
                per_depth_curvature=per_depth_curvature,
            )
            self._init_module_weights(self.hyperbolic_head.query_direction_proj)
            self._init_module_weights(self.hyperbolic_head.query_radius_proj)
            self._init_module_weights(self.hyperbolic_head.label_direction_proj)
            self._init_module_weights(self.hyperbolic_head.label_radius_proj)
            self._init_module_weights(self.hyperbolic_head.label_hyperplane_proj)
            self._init_module_weights(self.hyperbolic_head.label_hyperplane_bias_proj)
            radius_bias_init = math.log(math.expm1(0.05))
            self.hyperbolic_head.query_radius_proj.bias.data.fill_(radius_bias_init)
            self.hyperbolic_head.label_radius_proj.bias.data.fill_(radius_bias_init)
        if self.classifier_head == 'hybrid':
            alpha = min(max(float(self.hyperbolic_alpha), 1e-4), 1.0 - 1e-4)
            raw_alpha = math.log(alpha / (1.0 - alpha))
            alpha_shape = (self.num_depths,) if self.depth_aware_hyper_alpha else (1,)
            self.raw_hyperbolic_alpha_by_depth = nn.Parameter(
                torch.full(alpha_shape, raw_alpha, dtype=torch.float32)
            )
            self.raw_euclidean_branch_scale_by_depth = nn.Parameter(
                torch.zeros(self.num_depths, dtype=torch.float32)
            )
            self.raw_hyperbolic_branch_scale_by_depth = nn.Parameter(
                torch.zeros(self.num_depths, dtype=torch.float32)
            )
        self.register_buffer('special_token_ids', self._build_special_token_ids(), persistent=False)
        self.register_buffer('legacy_mlm_candidate_ids', self._build_legacy_mlm_candidate_ids(), persistent=False)
        self.register_buffer('mlm_candidate_ids', self._build_mlm_candidate_ids(), persistent=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        num_labels = kwargs.pop('num_labels')
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.num_labels = num_labels
        config.name_or_path = pretrained_model_name_or_path
        backbone_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, config=config)
        return cls(
            config=config,
            backbone_model=backbone_model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def _extract_backbone(self, backbone_model):
        base_prefix = getattr(backbone_model, 'base_model_prefix', None)
        if base_prefix is None or not hasattr(backbone_model, base_prefix):
            raise ValueError(f'Unsupported masked LM backbone: {backbone_model.__class__.__name__}')
        return getattr(backbone_model, base_prefix)

    def _extract_lm_head(self, backbone_model):
        for attr_name in ('cls', 'lm_head'):
            if hasattr(backbone_model, attr_name):
                return getattr(backbone_model, attr_name)
        raise ValueError(f'Unsupported masked LM head: {backbone_model.__class__.__name__}')

    def _decoder_accessor(self):
        if hasattr(self.cls, 'predictions') and hasattr(self.cls.predictions, 'decoder'):
            return self.cls.predictions, 'decoder'
        if hasattr(self.cls, 'decoder'):
            return self.cls, 'decoder'
        raise ValueError(f'Unsupported MLM head structure: {self.cls.__class__.__name__}')

    def _normalize_orth_method(self, orth_method):
        normalized = (orth_method or 'gram_schmidt').strip().lower()
        mapping = {
            'gram_schmidt': 'gram_schmidt',
            'gram-schmidt': 'gram_schmidt',
            'gs': 'gram_schmidt',
            'svd': 'svd',
            'householder': 'householder',
            'qr': 'householder',
            'none': 'none',
            'identity': 'none',
            'no_orth': 'none',
            'no-orth': 'none',
            'off': 'none',
        }
        if normalized not in mapping:
            supported = ', '.join(['gram_schmidt', 'svd', 'householder', 'none'])
            raise ValueError(f'Unsupported orthogonalization method: {orth_method}. Supported values: {supported}.')
        return mapping[normalized]

    def _normalize_mlm_mask_strategy(self, mlm_mask_strategy):
        normalized = (mlm_mask_strategy or 'legacy').strip().lower()
        mapping = {
            'legacy': 'legacy',
            'old': 'legacy',
            'legacy_bert': 'legacy_bert',
            'bert_legacy': 'legacy_bert',
            'legacy_tokenizer_aware': 'legacy_tokenizer_aware',
            'tokenizer_aware': 'legacy_tokenizer_aware',
            'legacy_safe': 'legacy_tokenizer_aware',
            'filtered': 'filtered',
            'safe': 'filtered',
            'current': 'filtered',
        }
        if normalized not in mapping:
            supported = ', '.join(['legacy', 'legacy_bert', 'legacy_tokenizer_aware', 'filtered'])
            raise ValueError(f'Unsupported MLM mask strategy: {mlm_mask_strategy}. Supported values: {supported}.')
        return mapping[normalized]

    def _normalize_classifier_head(self, classifier_head):
        normalized = (classifier_head or 'euclidean').strip().lower()
        mapping = {
            'euclidean': 'euclidean',
            'linear': 'euclidean',
            'euc': 'euclidean',
            'hyperbolic': 'hyperbolic',
            'hyp': 'hyperbolic',
            'hybrid': 'hybrid',
            'mix': 'hybrid',
        }
        if normalized not in mapping:
            supported = ', '.join(['euclidean', 'hyperbolic', 'hybrid'])
            raise ValueError(f'Unsupported classifier head: {classifier_head}. Supported values: {supported}.')
        return mapping[normalized]

    def _resolve_mlm_mask_strategy(self):
        if self.mlm_mask_strategy != 'legacy':
            return self.mlm_mask_strategy
        if getattr(self.config, 'model_type', '').lower() == 'bert':
            return 'legacy_bert'
        return 'legacy_tokenizer_aware'

    def get_effective_mlm_mask_strategy(self):
        return self._resolve_mlm_mask_strategy()

    def get_effective_classifier_head(self):
        return self.classifier_head

    def get_hyperbolic_alpha_by_depth(self):
        if self.classifier_head != 'hybrid':
            raise ValueError('Depth-aware alpha is only defined for the hybrid classifier head.')
        if hasattr(self, 'raw_hyperbolic_alpha_by_depth'):
            alpha = torch.sigmoid(self.raw_hyperbolic_alpha_by_depth)
            if alpha.numel() == 1:
                return alpha.expand(self.num_depths)
            return alpha
        return torch.full((self.num_depths,), float(self.hyperbolic_alpha), device=self.device, dtype=torch.float32)

    def get_euclidean_branch_scale_by_depth(self):
        if self.classifier_head != 'hybrid':
            raise ValueError('Euclidean branch scale is only defined for the hybrid classifier head.')
        return torch.exp(self.raw_euclidean_branch_scale_by_depth)

    def get_hyperbolic_branch_scale_by_depth(self):
        if self.classifier_head != 'hybrid':
            raise ValueError('Hyperbolic branch scale is only defined for the hybrid classifier head.')
        return torch.exp(self.raw_hyperbolic_branch_scale_by_depth)

    def _standardize_branch_logits(self, logits, eps=1e-6):
        branch_mean = logits.mean(dim=-1, keepdim=True)
        branch_std = logits.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (logits - branch_mean) / branch_std

    def _build_special_token_ids(self):
        special_ids = sorted(
            {
                int(token_id)
                for token_id in self.tokenizer.all_special_ids
                if token_id is not None and 0 <= int(token_id) < self.tokenizer.vocab_size
            }
        )
        return torch.tensor(special_ids, dtype=torch.long)

    def _build_legacy_mlm_candidate_ids(self):
        special_ids = set(self.special_token_ids.tolist())
        candidate_ids = [token_id for token_id in range(self.tokenizer.vocab_size) if token_id not in special_ids]
        if not candidate_ids:
            raise ValueError(f'No valid legacy MLM candidate tokens found for tokenizer: {self.name_or_path}')
        return torch.tensor(candidate_ids, dtype=torch.long)

    def _build_mlm_candidate_ids(self):
        special_ids = set(self._build_special_token_ids().tolist())
        candidate_ids = []
        for token_id in range(self.tokenizer.vocab_size):
            if token_id in special_ids:
                continue
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token, str) and token.startswith('[unused') and token.endswith(']'):
                continue
            candidate_ids.append(token_id)
        if not candidate_ids:
            raise ValueError(f'No valid MLM candidate tokens found for tokenizer: {self.name_or_path}')
        return torch.tensor(candidate_ids, dtype=torch.long)

    def _build_special_token_mask(self, input_ids):
        if self.special_token_ids.numel() == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_mask |= input_ids == token_id
        return special_mask

    def _apply_legacy_mlm_noise(self, input_ids, attention_mask, single_labels):
        enable_mask = input_ids < self.tokenizer.vocab_size
        random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
        input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
        random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
        mlm_mask = random_mask > 0.985
        input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
        single_labels = single_labels.masked_fill(random_mask < 0.85, -100)
        return input_ids, single_labels

    def _apply_legacy_tokenizer_aware_mlm_noise(self, input_ids, attention_mask, single_labels):
        enable_mask = input_ids < self.tokenizer.vocab_size
        random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
        input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
        random_token_indices = torch.randint(
            low=0,
            high=self.legacy_mlm_candidate_ids.numel(),
            size=input_ids.shape,
            device=input_ids.device,
        )
        random_ids = self.legacy_mlm_candidate_ids[random_token_indices]
        mlm_mask = random_mask > 0.985
        input_ids = torch.where(mlm_mask, random_ids, input_ids)
        single_labels = single_labels.masked_fill(random_mask < 0.85, -100)
        return input_ids, single_labels

    def _apply_filtered_mlm_noise(self, input_ids, attention_mask, single_labels):
        special_mask = self._build_special_token_mask(input_ids)
        enable_mask = (input_ids < self.tokenizer.vocab_size) & (~special_mask)
        random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
        input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
        random_token_indices = torch.randint(
            low=0,
            high=self.mlm_candidate_ids.numel(),
            size=input_ids.shape,
            device=input_ids.device,
        )
        random_ids = self.mlm_candidate_ids[random_token_indices]
        mlm_mask = random_mask > 0.985
        input_ids = torch.where(mlm_mask, random_ids, input_ids)
        single_labels = single_labels.masked_fill(random_mask < 0.85, -100)
        return input_ids, single_labels

    def _apply_mlm_noise(self, input_ids, attention_mask, single_labels):
        effective_strategy = self._resolve_mlm_mask_strategy()
        if effective_strategy == 'legacy_bert':
            return self._apply_legacy_mlm_noise(input_ids, attention_mask, single_labels)
        if effective_strategy == 'legacy_tokenizer_aware':
            return self._apply_legacy_tokenizer_aware_mlm_noise(input_ids, attention_mask, single_labels)
        if effective_strategy == 'filtered':
            return self._apply_filtered_mlm_noise(input_ids, attention_mask, single_labels)
        raise ValueError(f'Unsupported MLM mask strategy: {effective_strategy}')

    def _init_module_weights(self, module):
        initializer_range = getattr(self.config, 'initializer_range', 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.bert.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        decoder_parent, decoder_name = self._decoder_accessor()
        return getattr(decoder_parent, decoder_name)

    def set_output_embeddings(self, new_embeddings):
        decoder_parent, decoder_name = self._decoder_accessor()
        setattr(decoder_parent, decoder_name, new_embeddings)
        if hasattr(decoder_parent, 'bias'):
            decoder_parent.bias = new_embeddings.bias

    def _extract_multiclass_hidden(self, sequence_output, multiclass_pos):
        return sequence_output.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        ).view(-1, sequence_output.size(-1))

    def _extract_multiclass_hidden_by_depth(self, sequence_output, multiclass_pos):
        batch_size = sequence_output.size(0)
        multiclass_hidden = self._extract_multiclass_hidden(sequence_output, multiclass_pos)
        if batch_size == 0:
            return multiclass_hidden.view(0, 0, sequence_output.size(-1))
        if multiclass_hidden.size(0) % batch_size != 0:
            raise ValueError(
                f'Flattened multiclass hidden size {multiclass_hidden.size(0)} is not divisible by batch size {batch_size}.'
            )
        num_depths = multiclass_hidden.size(0) // batch_size
        return multiclass_hidden.reshape(batch_size, num_depths, sequence_output.size(-1))

    def _extract_multiclass_prediction_scores(self, prediction_scores, multiclass_pos):
        return prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))
        ).view(-1, prediction_scores.size(-1))

    def _get_label_classifier_embeddings(self):
        return self.get_output_embeddings().weight()[self.vocab_size:self.vocab_size + self.num_labels]

    def _compute_euclidean_multiclass_logits(self, prediction_scores, multiclass_pos):
        multiclass_scores = self._extract_multiclass_prediction_scores(prediction_scores, multiclass_pos)
        return multiclass_scores[:, self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias

    def _compute_hyperbolic_multiclass_outputs(self, sequence_output, multiclass_pos):
        if not hasattr(self, 'hyperbolic_head'):
            raise ValueError('Hyperbolic classifier head is not initialized.')
        multiclass_hidden_by_depth = self._extract_multiclass_hidden_by_depth(sequence_output, multiclass_pos)
        label_embeddings = self._get_label_classifier_embeddings()
        hyperbolic_outputs = self.hyperbolic_head.forward_with_aux(
            multiclass_hidden_by_depth,
            label_embeddings,
            self.label_depth_ids,
        )
        hyperbolic_outputs['query_hidden_by_depth'] = multiclass_hidden_by_depth
        hyperbolic_outputs['label_embeddings'] = label_embeddings
        return hyperbolic_outputs

    def _compute_hyperbolic_multiclass_logits(self, sequence_output, multiclass_pos, add_bias, hyperbolic_outputs=None):
        if hyperbolic_outputs is None:
            hyperbolic_outputs = self._compute_hyperbolic_multiclass_outputs(sequence_output, multiclass_pos)
        multiclass_logits = hyperbolic_outputs['logits']
        if add_bias:
            multiclass_logits = multiclass_logits + self.multiclass_bias
        return multiclass_logits

    def _compute_multiclass_logits(self, sequence_output, prediction_scores, multiclass_pos, hyperbolic_outputs=None):
        if self.classifier_head == 'euclidean':
            return self._compute_euclidean_multiclass_logits(prediction_scores, multiclass_pos)
        if self.classifier_head == 'hyperbolic':
            return self._compute_hyperbolic_multiclass_logits(
                sequence_output,
                multiclass_pos,
                add_bias=True,
                hyperbolic_outputs=hyperbolic_outputs,
            )
        if self.classifier_head == 'hybrid':
            euclidean_logits = self._compute_euclidean_multiclass_logits(prediction_scores, multiclass_pos)
            hyperbolic_logits = self._compute_hyperbolic_multiclass_logits(
                sequence_output,
                multiclass_pos,
                add_bias=False,
                hyperbolic_outputs=hyperbolic_outputs,
            )
            if hyperbolic_outputs is None:
                raise ValueError('hybrid classifier requires hyperbolic_outputs.')
            alpha_by_depth = self.get_hyperbolic_alpha_by_depth()
            euclidean_scale_by_depth = self.get_euclidean_branch_scale_by_depth()
            hyperbolic_scale_by_depth = self.get_hyperbolic_branch_scale_by_depth()
            batch_size = hyperbolic_outputs['query_hyp_by_depth'].size(0)
            euclidean_logits = euclidean_logits.view(batch_size, self.num_depths, self.num_labels)
            hyperbolic_logits = hyperbolic_logits.view(batch_size, self.num_depths, self.num_labels)
            euclidean_logits = self._standardize_branch_logits(euclidean_logits)
            hyperbolic_logits = self._standardize_branch_logits(hyperbolic_logits)
            return (
                euclidean_scale_by_depth.view(1, -1, 1) * euclidean_logits
                + alpha_by_depth.view(1, -1, 1)
                * hyperbolic_scale_by_depth.view(1, -1, 1)
                * hyperbolic_logits
            ).view(
                -1,
                self.num_labels,
            )
        raise ValueError(f'Unsupported classifier head: {self.classifier_head}')

    def _gram_schmidt(self, vectors):
        basis = []
        for vector in vectors:
            if basis:
                projection = sum(torch.dot(vector, base) / torch.dot(base, base) * base for base in basis)
                residual = vector - projection
            else:
                residual = vector
            if torch.linalg.norm(residual) > 1e-8:
                basis.append(residual / torch.linalg.norm(residual))

        if len(basis) != len(vectors):
            if vectors.size(0) > vectors.size(1):
                u, _, vh = torch.linalg.svd(vectors, full_matrices=False)
                return torch.matmul(u, vh)
            q_col, _ = torch.linalg.qr(vectors.transpose(0, 1), mode='reduced')
            return q_col.transpose(0, 1)

        return torch.stack(basis)

    def _align_signs_and_rescale(self, ortho_vectors, reference_vectors):
        aligned_vectors = ortho_vectors.clone()
        dots = torch.sum(aligned_vectors * reference_vectors, dim=1)
        signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
        aligned_vectors = aligned_vectors * signs.unsqueeze(1)
        target_norms = torch.linalg.norm(reference_vectors, dim=1, keepdim=True).clamp_min(1e-8)
        return aligned_vectors * target_norms

    def _orthogonalize_with_svd(self, vectors):
        u, _, vh = torch.linalg.svd(vectors, full_matrices=False)
        ortho_vectors = torch.matmul(u, vh)
        return self._align_signs_and_rescale(ortho_vectors, vectors)

    def _orthogonalize_with_householder(self, vectors):
        if vectors.size(0) > vectors.size(1):
            return self._orthogonalize_with_svd(vectors)
        q_col, _ = torch.linalg.qr(vectors.transpose(0, 1), mode='reduced')
        ortho_vectors = q_col.transpose(0, 1)
        return self._align_signs_and_rescale(ortho_vectors, vectors)

    def _orthogonalize(self, vectors):
        if self.orth_method == 'none':
            return vectors.clone()
        if self.orth_method == 'gram_schmidt':
            ortho_vectors = self._gram_schmidt(vectors)
            return self._align_signs_and_rescale(ortho_vectors, vectors)
        if self.orth_method == 'svd':
            return self._orthogonalize_with_svd(vectors)
        if self.orth_method == 'householder':
            return self._orthogonalize_with_householder(vectors)
        raise ValueError(f'Unsupported orthogonalization method: {self.orth_method}')

    def semantically_anchored_gof_initialization(self, initial_embeddings, depth2label, value2slot):
        structured_embeddings = initial_embeddings.clone()
        if self.orth_method == 'none':
            return structured_embeddings

        top_level_indices = depth2label.get(0, [])
        if len(top_level_indices) > 0:
            top_vectors_initial = structured_embeddings[top_level_indices]
            structured_embeddings[top_level_indices] = self._orthogonalize(top_vectors_initial)

        parent_to_children = {}
        for child, parent in value2slot.items():
            if parent == -1:
                continue
            parent_to_children.setdefault(parent, []).append(child)

        max_depth = max(depth2label.keys()) if depth2label else -1
        for depth in range(max_depth + 1):
            for parent_idx in depth2label.get(depth, []):
                children_indices = parent_to_children.get(parent_idx, [])
                if len(children_indices) <= 1:
                    continue

                parent_embedding = structured_embeddings[parent_idx]
                residuals = []
                for child_idx in children_indices:
                    child_embedding = structured_embeddings[child_idx]
                    projection = (
                        torch.dot(child_embedding, parent_embedding) / torch.dot(parent_embedding, parent_embedding)
                    ) * parent_embedding
                    residuals.append(child_embedding - projection)

                if not residuals:
                    continue

                residual_tensor = torch.stack(residuals)
                orthogonal_residuals = self._orthogonalize(residual_tensor)
                for index, child_idx in enumerate(children_indices):
                    child_initial = structured_embeddings[child_idx]
                    projection = (
                        torch.dot(child_initial, parent_embedding) / torch.dot(parent_embedding, parent_embedding)
                    ) * parent_embedding
                    structured_embeddings[child_idx] = projection + orthogonal_residuals[index]

        return structured_embeddings

    def _build_depth_prompt_embeddings(self, label_embeddings, depth):
        if depth <= 0:
            return label_embeddings.new_empty((0, label_embeddings.size(-1)))

        global_center = label_embeddings.mean(dim=0)
        global_norm = torch.linalg.norm(label_embeddings, dim=-1).mean().clamp_min(1e-8)
        depth_prompt_embeddings = []
        for depth_idx in range(depth):
            label_ids = self.depth2label.get(depth_idx, [])
            if label_ids:
                label_index = torch.tensor(label_ids, device=label_embeddings.device, dtype=torch.long)
                depth_label_embeddings = label_embeddings.index_select(0, label_index)
                center = depth_label_embeddings.mean(dim=0)
                target_norm = torch.linalg.norm(depth_label_embeddings, dim=-1).mean().clamp_min(1e-8)
            else:
                center = global_center
                target_norm = global_norm

            center_norm = torch.linalg.norm(center).clamp_min(1e-8)
            depth_prompt_embeddings.append(center / center_norm * target_norm)

        return torch.stack(depth_prompt_embeddings, dim=0)

    def init_embedding(self):
        depth = len(self.depth2label)
        tokenizer = self.tokenizer
        input_embeds = self.get_input_embeddings()
        label_emb_list = []

        if self.use_label_description:
            descriptions_path = os.path.join(self.data_path, 'label_descriptions.json')
            with open(descriptions_path, 'r', encoding='utf-8') as file_obj:
                label_descriptions = json.load(file_obj)
            label_descriptions = {int(k): v for k, v in label_descriptions.items()}
            label_tokens = {
                idx: tokenizer.encode(text, truncation=True, max_length=512)
                for idx, text in label_descriptions.items()
            }
            for label_id in range(self.num_labels):
                token_ids = label_tokens.get(label_id, [tokenizer.mask_token_id])
                token_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                label_emb_list.append(input_embeds.weight.index_select(0, token_tensor).mean(dim=0))
        else:
            label_name_map = _load_id2label_map(self.data_path, self.dataset_name)
            for label_id in range(self.num_labels):
                token_ids = tokenizer.encode(label_name_map[label_id], add_special_tokens=False)
                if not token_ids:
                    token_ids = [tokenizer.mask_token_id]
                token_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                label_emb_list.append(input_embeds.weight.index_select(0, token_tensor).mean(dim=0))

        initial_embeddings = torch.stack(label_emb_list)
        value2slot = {
            child: parent
            for parent, child in self.path_list
            if parent != -1 and child < self.num_labels and parent < self.num_labels
        }
        for label_id in range(self.num_labels):
            value2slot.setdefault(label_id, -1)

        ideal_label_embeddings = self.semantically_anchored_gof_initialization(
            initial_embeddings,
            self.depth2label,
            value2slot,
        )
        if hasattr(self, 'ideal_label_embeddings'):
            self.ideal_label_embeddings = ideal_label_embeddings.detach()
        else:
            self.register_buffer('ideal_label_embeddings', ideal_label_embeddings.detach())
        if hasattr(self, 'hyperbolic_head'):
            label_parent_indices = torch.tensor(
                [value2slot[label_id] for label_id in range(self.num_labels)],
                device=self.device,
                dtype=torch.long,
            )
            self.hyperbolic_head.initialize_label_structure(
                prototype_hidden=ideal_label_embeddings.detach(),
                label_depth_ids=self.label_depth_ids,
                label_parent_indices=label_parent_indices,
            )

        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id], device=self.device, dtype=torch.long))
        prompt_embedding = nn.Embedding(depth + 1, input_embeds.weight.size(1), padding_idx=0).to(self.device)
        self._init_module_weights(prompt_embedding)
        if not self.graph_type:
            depth_prompt_embeddings = self._build_depth_prompt_embeddings(ideal_label_embeddings, depth)
            with torch.no_grad():
                prompt_embedding.weight[1:, :].copy_(depth_prompt_embeddings)

        label_emb = torch.cat([ideal_label_embeddings, prompt_embedding.weight[1:, :], prefix], dim=0)
        node_depth_ids = torch.cat(
            [
                self.label_depth_ids.detach().cpu(),
                torch.arange(depth, dtype=torch.long),
            ],
            dim=0,
        )
        embedding = GraphEmbedding(
            self.config,
            input_embeds,
            label_emb,
            self.graph_type,
            path_list=self.path_list,
            layer=self.layer,
            data_path=self.data_path,
            graph_space=self.graph_space,
            node_depth_ids=node_depth_ids,
            num_depths=self.num_depths,
            curvature_init=self.hyperbolic_curvature_init,
            curvature_init_by_depth=self.curvature_init_by_depth,
            per_depth_curvature=self.per_depth_curvature,
            tangent_clip=self.hyperbolic_radius_clip,
        )
        self.set_input_embeddings(embedding)

        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = F.pad(
            output_embeddings.bias.data,
            (0, embedding.size - output_embeddings.bias.shape[0]),
            "constant",
            0,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)

        if self.training:
            input_ids, single_labels = self._apply_mlm_noise(input_ids, attention_mask, single_labels)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        hyperbolic_outputs = None
        if hasattr(self, 'hyperbolic_head'):
            hyperbolic_outputs = self._compute_hyperbolic_multiclass_outputs(sequence_output, multiclass_pos)
        multiclass_logits = self._compute_multiclass_logits(
            sequence_output,
            prediction_scores,
            multiclass_pos,
            hyperbolic_outputs=hyperbolic_outputs,
        )

        masked_lm_loss = None
        total_loss = None
        multiclass_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                single_labels.view(-1),
            )
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            total_loss = masked_lm_loss + multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        result = MaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        result.multiclass_logits = multiclass_logits
        if hyperbolic_outputs is not None:
            result.hyperbolic_logits = hyperbolic_outputs['logits']
            result.hyperbolic_logits_by_depth = hyperbolic_outputs['logits_by_depth']
            result.hyperbolic_raw_query_tangent = hyperbolic_outputs['raw_query_tangent']
            result.hyperbolic_raw_query_tangent_by_depth = hyperbolic_outputs['raw_query_tangent_by_depth']
            result.hyperbolic_query_tangent = hyperbolic_outputs['query_tangent']
            result.hyperbolic_query_tangent_by_depth = hyperbolic_outputs['query_tangent_by_depth']
            result.hyperbolic_query_radius_by_depth = hyperbolic_outputs['query_radius_by_depth']
            result.hyperbolic_query_radius_floor_by_depth = hyperbolic_outputs['query_radius_floor_by_depth']
            result.hyperbolic_raw_query_radius_residual_by_depth = hyperbolic_outputs['raw_query_radius_residual_by_depth']
            result.hyperbolic_raw_query_radius_by_depth = hyperbolic_outputs['raw_query_radius_by_depth']
            result.hyperbolic_raw_query_radius_delta_by_depth = hyperbolic_outputs['raw_query_radius_delta_by_depth']
            result.hyperbolic_raw_query_direction_by_depth = hyperbolic_outputs['raw_query_direction_by_depth']
            result.hyperbolic_query_direction_by_depth = hyperbolic_outputs['query_direction_by_depth']
            result.hyperbolic_raw_label_tangent = hyperbolic_outputs['raw_label_tangent']
            result.hyperbolic_label_tangent = hyperbolic_outputs['label_tangent']
            result.hyperbolic_raw_label_radius = hyperbolic_outputs['raw_label_radius']
            result.hyperbolic_label_radius = hyperbolic_outputs['label_radius']
            result.hyperbolic_label_radius_floor = hyperbolic_outputs['label_radius_floor']
            result.hyperbolic_raw_label_radius_residual = hyperbolic_outputs['raw_label_radius_residual']
            result.hyperbolic_raw_label_direction = hyperbolic_outputs['raw_label_direction']
            result.hyperbolic_label_direction = hyperbolic_outputs['label_direction']
            result.hyperbolic_query_points = hyperbolic_outputs['query_hyp']
            result.hyperbolic_query_points_by_depth = hyperbolic_outputs['query_hyp_by_depth']
            result.hyperbolic_label_points = hyperbolic_outputs['label_hyp']
            result.hyperbolic_semantic_label_tangent = hyperbolic_outputs['semantic_label_tangent']
            result.hyperbolic_semantic_label_points = hyperbolic_outputs['semantic_label_hyp']
            result.hyperbolic_alignment_label_tangent = hyperbolic_outputs['alignment_label_tangent']
            result.hyperbolic_alignment_label_points = hyperbolic_outputs['alignment_label_hyp']
            result.hyperbolic_label_hyperplane_raw_space = hyperbolic_outputs['label_hyperplane_raw_space']
            result.hyperbolic_label_hyperplane_space = hyperbolic_outputs['label_hyperplane_space']
            result.hyperbolic_label_hyperplane_time = hyperbolic_outputs['label_hyperplane_time']
            result.hyperbolic_label_hyperplane_bias = hyperbolic_outputs['label_hyperplane_bias']
            result.hyperbolic_label_hyperplane_scale = hyperbolic_outputs['label_hyperplane_scale']
            result.hyperbolic_curvature = hyperbolic_outputs['curvature']
            result.hyperbolic_curvature_by_depth = hyperbolic_outputs['curvature']
            result.hyperbolic_label_curvature = hyperbolic_outputs['label_curvature']
            result.hyperbolic_logit_scale = hyperbolic_outputs['logit_scale']
            if self.classifier_head == 'hybrid':
                result.hyperbolic_alpha_by_depth = self.get_hyperbolic_alpha_by_depth()
                result.euclidean_branch_scale_by_depth = self.get_euclidean_branch_scale_by_depth()
                result.hyperbolic_branch_scale_by_depth = self.get_hyperbolic_branch_scale_by_depth()
        return result, masked_lm_loss, multiclass_loss

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs, masked_lm_loss, multiclass_loss = self(input_ids, attention_mask)
        prediction_scores = outputs.multiclass_logits.view(-1, len(depth2label), self.num_labels)

        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for depth, score in enumerate(scores):
                for label_id in depth2label[depth]:
                    if score[label_id] > 0:
                        predict_labels[-1].append(label_id)
        return predict_labels, prediction_scores
