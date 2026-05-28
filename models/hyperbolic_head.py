import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz_ops import exp_map0, lorentz_time, point_to_hyperplane_scores, smooth_clip_tangent_norm


def _inverse_softplus(value):
    if value <= 0:
        raise ValueError(f'curvature_init must be positive, got {value}')
    return math.log(math.expm1(value))


def _inverse_softplus_tensor(value):
    value = value.clamp_min(1e-6)
    return torch.log(torch.expm1(value))


def _parse_curvature_init(curvature_init, curvature_init_by_depth, num_depths):
    if curvature_init_by_depth is None:
        return [float(curvature_init)] * num_depths
    if isinstance(curvature_init_by_depth, str):
        values = [float(item.strip()) for item in curvature_init_by_depth.split(',') if item.strip()]
    else:
        values = [float(item) for item in curvature_init_by_depth]
    if not values:
        return [float(curvature_init)] * num_depths
    if len(values) == 1:
        return values * num_depths
    if len(values) != num_depths:
        raise ValueError(
            f'curvature_init_by_depth length {len(values)} must be 1 or equal to num_depths {num_depths}.'
        )
    return values


class HyperbolicClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hyperbolic_dim=256,
        num_depths=1,
        curvature_init=1.0,
        curvature_init_by_depth=None,
        logit_scale_init=1.0,
        tangent_clip=2.0,
        per_depth_curvature=False,
    ):
        super().__init__()
        self.query_input_norm = nn.LayerNorm(input_dim)
        self.label_input_norm = nn.LayerNorm(input_dim)
        self.query_direction_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.query_radius_proj = nn.Linear(input_dim, 1)
        self.label_direction_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.label_radius_proj = nn.Linear(input_dim, 1)
        self.label_hyperplane_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.label_hyperplane_bias_proj = nn.Linear(input_dim, 1)
        self.hyperbolic_dim = hyperbolic_dim
        self.num_depths = max(int(num_depths), 1)
        self.per_depth_curvature = per_depth_curvature

        init_values = _parse_curvature_init(curvature_init, curvature_init_by_depth, self.num_depths)
        raw_init = [_inverse_softplus(value) for value in init_values]
        if self.per_depth_curvature:
            raw_tensor = torch.tensor(raw_init, dtype=torch.float32)
        else:
            raw_tensor = torch.tensor([raw_init[0]], dtype=torch.float32)
        self.raw_curvature_by_depth = nn.Parameter(raw_tensor)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(max(logit_scale_init, 1e-4)), dtype=torch.float32))
        self.tangent_clip = tangent_clip
        # Keep path supervision closer to the live label semantics while still
        # respecting the structural prototype branch.
        self.alignment_semantic_weight = 0.7
        self.register_buffer(
            'label_radius_floor_by_depth',
            self._build_label_radius_floor_schedule(),
            persistent=False,
        )
        self.register_buffer(
            'query_radius_floor_by_depth',
            self._build_query_radius_floor_schedule(),
            persistent=False,
        )
        self.label_structure_direction = None
        self.label_structure_radius_residual = None
        self.register_buffer(
            'label_structure_depth_ids',
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    def _ensure_depth_ids(self, depth_ids, reference):
        if depth_ids is None:
            raise ValueError('depth_ids must be provided when using depth-aware hyperbolic projections.')
        if not isinstance(depth_ids, torch.Tensor):
            depth_ids = torch.tensor(depth_ids, device=reference.device, dtype=torch.long)
        return depth_ids.to(device=reference.device, dtype=torch.long)

    def get_all_curvatures(self):
        base = F.softplus(self.raw_curvature_by_depth) + 1e-4
        if self.per_depth_curvature:
            return base
        return base.expand(self.num_depths)

    def get_curvature_by_depth(self, depth_ids):
        depth_ids = self._ensure_depth_ids(depth_ids, self.raw_curvature_by_depth)
        all_curvatures = self.get_all_curvatures()
        return all_curvatures.index_select(0, depth_ids)

    def get_logit_scale(self):
        return torch.exp(self.logit_scale)

    def _clip_tangent_norm(self, vectors):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return vectors
        return smooth_clip_tangent_norm(vectors, max_norm=self.tangent_clip)

    def _build_label_radius_floor_schedule(self):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return torch.zeros(self.num_depths, dtype=torch.float32)
        if self.num_depths == 1:
            return torch.tensor([0.2 * float(self.tangent_clip)], dtype=torch.float32)
        max_radius = float(self.tangent_clip)
        # Keep shallow labels away from the origin, while leaving room for per-label residuals.
        return torch.linspace(0.1 * max_radius, 0.45 * max_radius, self.num_depths, dtype=torch.float32)

    def _build_query_radius_floor_schedule(self):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return torch.zeros(self.num_depths, dtype=torch.float32)
        if self.num_depths == 1:
            return torch.tensor([0.08 * float(self.tangent_clip)], dtype=torch.float32)
        max_radius = float(self.tangent_clip)
        # Keep the query chain away from the origin and force a coarse-to-fine outward trend.
        return torch.linspace(0.05 * max_radius, 0.35 * max_radius, self.num_depths, dtype=torch.float32)

    def _bound_residual_radius(self, raw_residual_radius, base_radius):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return raw_residual_radius
        max_radius = torch.as_tensor(self.tangent_clip, device=raw_residual_radius.device, dtype=raw_residual_radius.dtype)
        available_radius = (max_radius - base_radius).clamp_min(1e-6)
        return available_radius * (1.0 - torch.exp(-raw_residual_radius / available_radius))

    def _target_radius_to_residual(self, target_radius, base_radius):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return (target_radius - base_radius).clamp_min(1e-6)

        max_radius = torch.as_tensor(self.tangent_clip, device=target_radius.device, dtype=target_radius.dtype)
        available_radius = (max_radius - base_radius).clamp_min(1e-6)
        bounded_ratio = ((target_radius - base_radius) / available_radius).clamp(min=0.0, max=1.0 - 1e-6)
        return -available_radius * torch.log1p(-bounded_ratio)

    def _build_structure_directions(self, prototype_hidden):
        num_labels, hidden_dim = prototype_hidden.shape
        centered_hidden = prototype_hidden - prototype_hidden.mean(dim=0, keepdim=True)
        reduced_dim = min(num_labels, hidden_dim, self.hyperbolic_dim)

        if reduced_dim > 0:
            try:
                _, _, vh = torch.linalg.svd(centered_hidden, full_matrices=False)
                basis = vh[:reduced_dim].transpose(0, 1)
                reduced_hidden = centered_hidden @ basis
            except RuntimeError:
                reduced_hidden = centered_hidden[:, :reduced_dim]
        else:
            reduced_hidden = centered_hidden.new_zeros((num_labels, 0))

        direction = centered_hidden.new_zeros((num_labels, self.hyperbolic_dim))
        if reduced_dim > 0:
            direction[:, :reduced_dim] = reduced_hidden

        direction_norm = torch.norm(direction, dim=-1, keepdim=True)
        zero_mask = direction_norm.squeeze(-1) < 1e-6
        if zero_mask.any():
            fallback = torch.zeros_like(direction[zero_mask])
            fallback_rows = torch.arange(fallback.size(0), device=direction.device) % max(self.hyperbolic_dim, 1)
            fallback[torch.arange(fallback.size(0), device=direction.device), fallback_rows] = 1.0
            direction[zero_mask] = fallback

        return F.normalize(direction, dim=-1, eps=1e-6)

    def _compute_subtree_sizes(self, label_parent_indices):
        parent_list = label_parent_indices.detach().cpu().tolist()
        num_labels = len(parent_list)
        children_by_parent = [[] for _ in range(num_labels)]
        roots = []

        for child_idx, parent_idx in enumerate(parent_list):
            if parent_idx is None or parent_idx < 0:
                roots.append(child_idx)
                continue
            if parent_idx < num_labels:
                children_by_parent[parent_idx].append(child_idx)

        subtree_sizes = [0] * num_labels

        def dfs(node_idx):
            total = 1
            for child_idx in children_by_parent[node_idx]:
                total += dfs(child_idx)
            subtree_sizes[node_idx] = total
            return total

        for root_idx in roots:
            dfs(root_idx)

        return torch.tensor(
            subtree_sizes,
            device=label_parent_indices.device,
            dtype=torch.float32,
        )

    def _build_structure_target_radii(self, label_depth_ids, label_parent_indices, dtype):
        base_radius = self.label_radius_floor_by_depth.index_select(0, label_depth_ids).to(dtype=dtype)
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return base_radius + 0.1

        subtree_sizes = self._compute_subtree_sizes(label_parent_indices).to(device=label_depth_ids.device, dtype=dtype)
        subtree_scores = torch.log1p(subtree_sizes)
        balance_scores = torch.full_like(subtree_scores, 0.5)

        for depth_idx in range(self.num_depths):
            depth_mask = label_depth_ids == depth_idx
            if not depth_mask.any():
                continue
            depth_scores = subtree_scores[depth_mask]
            depth_span = depth_scores.max() - depth_scores.min()
            if float(depth_span) < 1e-6:
                balance_scores[depth_mask] = 0.5
                continue
            # Larger subtree => stay more inward; leaves/small branches can move outward.
            balance_scores[depth_mask] = (depth_scores.max() - depth_scores) / depth_span.clamp_min(1e-6)

        depth_progress = label_depth_ids.to(dtype=dtype)
        if self.num_depths > 1:
            depth_progress = depth_progress / float(self.num_depths - 1)
        else:
            depth_progress = depth_progress.new_zeros(depth_progress.shape)

        available_radius = (
            torch.as_tensor(self.tangent_clip, device=label_depth_ids.device, dtype=dtype) - base_radius
        ).clamp_min(1e-6)
        # Start label prototypes much closer to the v4 radius regime. The previous
        # schedule pushed deep labels too far outward and made the cone almost
        # entirely infeasible from the first epoch.
        residual_fraction = (0.02 + 0.04 * depth_progress + 0.05 * balance_scores).clamp(min=0.02, max=0.16)
        return base_radius + available_radius * residual_fraction

    def initialize_label_structure(self, prototype_hidden, label_depth_ids, label_parent_indices):
        if prototype_hidden.dim() != 2:
            raise ValueError(
                f'prototype_hidden must have shape [num_labels, hidden], got {tuple(prototype_hidden.shape)}'
            )

        label_depth_ids = self._ensure_depth_ids(label_depth_ids, prototype_hidden)
        label_parent_indices = label_parent_indices.to(device=prototype_hidden.device, dtype=torch.long)
        initial_direction = self._build_structure_directions(prototype_hidden.detach())
        target_radius = self._build_structure_target_radii(
            label_depth_ids=label_depth_ids,
            label_parent_indices=label_parent_indices,
            dtype=prototype_hidden.dtype,
        )
        base_radius = self.label_radius_floor_by_depth.index_select(0, label_depth_ids).to(
            device=prototype_hidden.device,
            dtype=prototype_hidden.dtype,
        )
        target_residual = self._target_radius_to_residual(target_radius, base_radius)
        raw_radius_residual = _inverse_softplus_tensor(target_residual)

        self.label_structure_direction = nn.Parameter(initial_direction.detach().clone())
        self.label_structure_radius_residual = nn.Parameter(raw_radius_residual.detach().clone())
        self.label_structure_depth_ids = label_depth_ids.detach().clone()

    def has_label_structure(self):
        return self.label_structure_direction is not None and self.label_structure_radius_residual is not None

    def _build_alignment_label_projection(self, structure_projection, semantic_projection):
        if structure_projection is None:
            return semantic_projection

        semantic_weight = float(self.alignment_semantic_weight)
        structure_weight = 1.0 - semantic_weight
        alignment_tangent = (
            semantic_weight * semantic_projection['tangent']
            + structure_weight * structure_projection['tangent'].detach()
        )
        alignment_curvature = semantic_projection['curvature'].unsqueeze(-1)
        alignment_hyp = exp_map0(alignment_tangent, curv=alignment_curvature)
        alignment_radius = torch.norm(alignment_tangent, dim=-1)
        return {
            'tangent': alignment_tangent,
            'points': alignment_hyp,
            'radius': alignment_radius,
            'curvature': semantic_projection['curvature'],
        }

    def _project_label_structure(self, label_depth_ids):
        if not self.has_label_structure():
            raise ValueError('label structure prototypes have not been initialized.')

        depth_ids = self._ensure_depth_ids(label_depth_ids, self.label_structure_direction)
        curvature = self.get_curvature_by_depth(depth_ids).unsqueeze(-1)
        direction = F.normalize(self.label_structure_direction, dim=-1, eps=1e-6)
        base_radius = self.label_radius_floor_by_depth.index_select(0, depth_ids).to(
            device=direction.device,
            dtype=direction.dtype,
        )
        raw_radius_residual = F.softplus(self.label_structure_radius_residual)
        bounded_radius = base_radius + self._bound_residual_radius(raw_radius_residual, base_radius)
        raw_radius = base_radius + raw_radius_residual
        raw_tangent = direction * raw_radius.unsqueeze(-1)
        tangent = direction * bounded_radius.unsqueeze(-1)
        hyp = exp_map0(tangent, curv=curvature)
        return {
            'raw_direction': self.label_structure_direction,
            'direction': direction,
            'base_radius': base_radius,
            'raw_radius_residual': raw_radius_residual,
            'raw_radius': raw_radius,
            'radius': bounded_radius,
            'raw_tangent': raw_tangent,
            'tangent': tangent,
            'points': hyp,
            'curvature': curvature.squeeze(-1),
        }

    def _project_label(self, hidden, depth_ids):
        depth_ids = self._ensure_depth_ids(depth_ids, hidden)
        curvature = self.get_curvature_by_depth(depth_ids).unsqueeze(-1)
        normalized_hidden = self.label_input_norm(hidden)
        raw_direction = torch.tanh(self.label_direction_proj(normalized_hidden))
        direction = F.normalize(raw_direction, dim=-1, eps=1e-6)
        base_radius = self.label_radius_floor_by_depth.index_select(0, depth_ids).to(device=hidden.device, dtype=hidden.dtype)
        raw_radius_residual = F.softplus(self.label_radius_proj(normalized_hidden).squeeze(-1))
        bounded_radius = base_radius + self._bound_residual_radius(raw_radius_residual, base_radius)
        raw_radius = base_radius + raw_radius_residual
        raw_tangent = direction * raw_radius.unsqueeze(-1)
        tangent = direction * bounded_radius.unsqueeze(-1)
        hyp = exp_map0(tangent, curv=curvature)
        return {
            'raw_direction': raw_direction,
            'direction': direction,
            'base_radius': base_radius,
            'raw_radius_residual': raw_radius_residual,
            'raw_radius': raw_radius,
            'radius': bounded_radius,
            'raw_tangent': raw_tangent,
            'tangent': tangent,
            'points': hyp,
            'normalized_hidden': normalized_hidden,
            'curvature': curvature.squeeze(-1),
        }

    def _bound_query_radius(self, raw_radius_by_depth):
        if self.tangent_clip is None or self.tangent_clip <= 0:
            return raw_radius_by_depth

        max_radius = torch.as_tensor(self.tangent_clip, device=raw_radius_by_depth.device, dtype=raw_radius_by_depth.dtype)
        max_radius = max_radius.clamp_min(1e-6)
        return max_radius * (1.0 - torch.exp(-raw_radius_by_depth / max_radius))

    def _project_query(self, query_hidden_by_depth):
        if query_hidden_by_depth.dim() != 3:
            raise ValueError(
                f'query_hidden_by_depth must have shape [batch, num_depths, hidden], got {tuple(query_hidden_by_depth.shape)}'
            )

        batch_size, num_depths, hidden_dim = query_hidden_by_depth.shape
        if num_depths != self.num_depths:
            raise ValueError(f'Expected num_depths {self.num_depths}, got {num_depths}.')

        depth_ids = torch.arange(num_depths, device=query_hidden_by_depth.device, dtype=torch.long)
        depth_ids = depth_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        curvature = self.get_curvature_by_depth(depth_ids).reshape(batch_size, num_depths)

        flat_hidden = query_hidden_by_depth.reshape(-1, hidden_dim)
        normalized_hidden = self.query_input_norm(flat_hidden)
        raw_direction = torch.tanh(self.query_direction_proj(normalized_hidden)).reshape(batch_size, num_depths, -1)
        direction = F.normalize(raw_direction, dim=-1, eps=1e-6)

        raw_radius_delta = F.softplus(self.query_radius_proj(normalized_hidden).reshape(batch_size, num_depths))
        raw_radius_residual = torch.cumsum(raw_radius_delta, dim=1)
        query_radius_floor = self.query_radius_floor_by_depth.to(
            device=query_hidden_by_depth.device,
            dtype=query_hidden_by_depth.dtype,
        ).unsqueeze(0).expand(batch_size, -1)
        bounded_radius = query_radius_floor + self._bound_residual_radius(raw_radius_residual, query_radius_floor)
        raw_radius = query_radius_floor + raw_radius_residual

        raw_tangent = direction * raw_radius.unsqueeze(-1)
        tangent = direction * bounded_radius.unsqueeze(-1)
        hyp = exp_map0(tangent.reshape(-1, self.hyperbolic_dim), curv=curvature.reshape(-1, 1)).reshape(
            batch_size,
            num_depths,
            self.hyperbolic_dim,
        )

        return {
            'raw_direction_by_depth': raw_direction,
            'direction_by_depth': direction,
            'raw_radius_delta_by_depth': raw_radius_delta,
            'radius_floor_by_depth': query_radius_floor,
            'raw_radius_residual_by_depth': raw_radius_residual,
            'raw_radius_by_depth': raw_radius,
            'bounded_radius_by_depth': bounded_radius,
            'raw_tangent_by_depth': raw_tangent,
            'tangent_by_depth': tangent,
            'points_by_depth': hyp,
            'curvature_by_depth': curvature,
        }

    def _build_label_hyperplanes(self, label_hidden, label_curvature):
        # Decouple classifier hyperplanes from structural label points so cone/path
        # can shape the hierarchy without being directly overwritten by classifier updates.
        normalized_hidden = self.label_input_norm(label_hidden)
        raw_normal_space = self.label_hyperplane_proj(normalized_hidden)
        normal_space = self._clip_tangent_norm(raw_normal_space)
        normal_time = lorentz_time(normal_space, curv=label_curvature.unsqueeze(-1)).squeeze(-1)
        bias = self.label_hyperplane_bias_proj(normalized_hidden).squeeze(-1)
        return {
            'raw_normal_space': raw_normal_space,
            'normal_space': normal_space,
            'normal_time': normal_time,
            'bias': bias,
            'scale': torch.rsqrt(label_curvature.clamp_min(1e-8)),
        }

    def project_query_by_depth(self, query_hidden_by_depth):
        query_projection = self._project_query(query_hidden_by_depth)
        return (
            query_projection['raw_tangent_by_depth'],
            query_projection['tangent_by_depth'],
            query_projection['points_by_depth'],
            query_projection['curvature_by_depth'],
        )

    def project_label_by_depth(self, label_hidden, label_depth_ids):
        return self._project_label(label_hidden, depth_ids=label_depth_ids)

    def compute_logits(self, query_hyp_by_depth, label_hyperplanes, label_depth_ids):
        batch_size, num_depths, _ = query_hyp_by_depth.shape
        label_depth_ids = self._ensure_depth_ids(label_depth_ids, query_hyp_by_depth)
        curvature_by_depth = self.get_all_curvatures()
        logits_by_depth = query_hyp_by_depth.new_zeros((batch_size, num_depths, label_depth_ids.size(0)))

        for depth_idx in range(num_depths):
            label_mask = label_depth_ids == depth_idx
            if not label_mask.any():
                continue
            depth_logits = self.get_logit_scale() * point_to_hyperplane_scores(
                query_hyp_by_depth[:, depth_idx, :],
                label_hyperplanes['normal_space'][label_mask],
                label_hyperplanes['normal_time'][label_mask],
                label_hyperplanes['bias'][label_mask],
                curv=curvature_by_depth[depth_idx],
            )
            logits_by_depth[:, depth_idx, label_mask] = depth_logits

        return logits_by_depth

    def forward_with_aux(self, query_hidden_by_depth, label_hidden, label_depth_ids):
        query_projection = self._project_query(query_hidden_by_depth)
        raw_query_tangent_by_depth = query_projection['raw_tangent_by_depth']
        query_tangent_by_depth = query_projection['tangent_by_depth']
        query_hyp_by_depth = query_projection['points_by_depth']
        query_curvature_by_depth = query_projection['curvature_by_depth']
        semantic_label_projection = self.project_label_by_depth(label_hidden, label_depth_ids)
        if self.has_label_structure():
            structure_label_projection = self._project_label_structure(label_depth_ids)
            alignment_label_projection = self._build_alignment_label_projection(
                structure_label_projection,
                semantic_label_projection,
            )
            label_projection = structure_label_projection
        else:
            structure_label_projection = None
            alignment_label_projection = semantic_label_projection
            label_projection = semantic_label_projection
        raw_label_tangent = label_projection['raw_tangent']
        label_tangent = label_projection['tangent']
        label_hyp = label_projection['points']
        label_curvature = label_projection['curvature']
        label_hyperplanes = self._build_label_hyperplanes(label_hidden, label_curvature)
        logits_by_depth = self.compute_logits(query_hyp_by_depth, label_hyperplanes, label_depth_ids)

        return {
            'raw_query_tangent': raw_query_tangent_by_depth.reshape(-1, raw_query_tangent_by_depth.size(-1)),
            'raw_query_tangent_by_depth': raw_query_tangent_by_depth,
            'query_tangent': query_tangent_by_depth.reshape(-1, query_tangent_by_depth.size(-1)),
            'query_tangent_by_depth': query_tangent_by_depth,
            'query_radius_by_depth': query_projection['bounded_radius_by_depth'],
            'query_radius_floor_by_depth': query_projection['radius_floor_by_depth'],
            'raw_query_radius_residual_by_depth': query_projection['raw_radius_residual_by_depth'],
            'raw_query_radius_by_depth': query_projection['raw_radius_by_depth'],
            'raw_query_radius_delta_by_depth': query_projection['raw_radius_delta_by_depth'],
            'raw_query_direction_by_depth': query_projection['raw_direction_by_depth'],
            'query_direction_by_depth': query_projection['direction_by_depth'],
            'raw_label_tangent': raw_label_tangent,
            'label_tangent': label_tangent,
            'raw_label_radius': label_projection['raw_radius'],
            'label_radius': label_projection['radius'],
            'label_radius_floor': label_projection['base_radius'],
            'raw_label_radius_residual': label_projection['raw_radius_residual'],
            'raw_label_direction': label_projection['raw_direction'],
            'label_direction': label_projection['direction'],
            'query_hyp': query_hyp_by_depth.reshape(-1, query_hyp_by_depth.size(-1)),
            'query_hyp_by_depth': query_hyp_by_depth,
            'label_hyp': label_hyp,
            'semantic_label_tangent': semantic_label_projection['tangent'],
            'semantic_label_hyp': semantic_label_projection['points'],
            'alignment_label_tangent': alignment_label_projection['tangent'],
            'alignment_label_hyp': alignment_label_projection['points'],
            'label_hyperplane_raw_space': label_hyperplanes['raw_normal_space'],
            'label_hyperplane_space': label_hyperplanes['normal_space'],
            'label_hyperplane_time': label_hyperplanes['normal_time'],
            'label_hyperplane_bias': label_hyperplanes['bias'],
            'label_hyperplane_scale': label_hyperplanes['scale'],
            'curvature': self.get_all_curvatures(),
            'query_curvature_by_depth': query_curvature_by_depth,
            'label_curvature': label_curvature,
            'logit_scale': self.get_logit_scale(),
            'logits': logits_by_depth.reshape(-1, logits_by_depth.size(-1)),
            'logits_by_depth': logits_by_depth,
        }

    def forward(self, query_hidden_by_depth, label_hidden, label_depth_ids):
        return self.forward_with_aux(query_hidden_by_depth, label_hidden, label_depth_ids)['logits']
