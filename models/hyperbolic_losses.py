import math

import torch
import torch.nn.functional as F

from .lorentz_ops import exp_map0, hyperbolic_float32, pairwise_dist


def _as_curvature_tensor(curv, reference):
    return torch.as_tensor(curv, device=reference.device, dtype=reference.dtype).clamp_min(1e-8)


def _resolve_depth_curvature(curvatures, depth_ids, reference):
    if isinstance(curvatures, torch.Tensor) and curvatures.dim() > 0:
        depth_ids = depth_ids.to(device=curvatures.device, dtype=torch.long)
        resolved = curvatures.index_select(0, depth_ids)
        return resolved.to(device=reference.device, dtype=reference.dtype)
    return _as_curvature_tensor(curvatures, reference).expand(depth_ids.size(0))


@hyperbolic_float32
def lorentz_half_aperture(points, curv=1.0, min_radius=0.1, eps=1e-6):
    curv = _as_curvature_tensor(curv, points)
    safe_min_radius = max(float(min_radius), eps)
    point_norm = torch.norm(points, dim=-1)
    asin_input = 2.0 * safe_min_radius / (torch.sqrt(curv) * point_norm + eps)
    return torch.asin(torch.clamp(asin_input, min=-1.0 + eps, max=1.0 - eps))


@hyperbolic_float32
def lorentz_oxy_angle(parents, children, curv=1.0, eps=1e-6):
    if parents.shape != children.shape:
        raise ValueError(
            f'parents.shape {tuple(parents.shape)} must match children.shape {tuple(children.shape)}'
        )

    curv = _as_curvature_tensor(curv, parents)
    parent_time = torch.sqrt(torch.clamp(1.0 / curv + torch.sum(parents ** 2, dim=-1), min=eps))
    child_time = torch.sqrt(torch.clamp(1.0 / curv + torch.sum(children ** 2, dim=-1), min=eps))
    c_parent_child = curv * (torch.sum(parents * children, dim=-1) - parent_time * child_time)

    acos_numer = child_time + c_parent_child * parent_time
    acos_denom = torch.sqrt(torch.clamp(c_parent_child ** 2 - 1.0, min=eps))
    parent_norm = torch.norm(parents, dim=-1)
    acos_input = acos_numer / (parent_norm * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1.0 + eps, max=1.0 - eps))


@hyperbolic_float32
def exact_entailment_cone_violation(parent_points, child_points, curv=1.0, min_radius=0.1, margin=0.0, eps=1e-6):
    angle = lorentz_oxy_angle(parent_points, child_points, curv=curv, eps=eps)
    aperture = lorentz_half_aperture(parent_points, curv=curv, min_radius=min_radius, eps=eps)
    violation = torch.clamp(angle - aperture + margin, min=0.0)
    return violation, angle, aperture


@hyperbolic_float32
def lorentz_distance(x, y, curv=1.0, eps=1e-6):
    if x.shape != y.shape:
        raise ValueError(f'x.shape {tuple(x.shape)} must match y.shape {tuple(y.shape)}')

    curv = _as_curvature_tensor(curv, x)
    x_time = torch.sqrt(torch.clamp(1.0 / curv + torch.sum(x ** 2, dim=-1), min=eps))
    y_time = torch.sqrt(torch.clamp(1.0 / curv + torch.sum(y ** 2, dim=-1), min=eps))
    c_xy = -curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    return torch.acosh(torch.clamp(c_xy, min=1.0 + eps)) / torch.sqrt(curv)


def label_cone_loss(
    label_points,
    parent_indices,
    child_indices,
    curv,
    margin=0.0,
    min_radius=0.1,
    eps=1e-6,
    hard_pool_ratio=1.0,
    smooth_band=0.05,
):
    zero = label_points.new_zeros(())
    if parent_indices is None or child_indices is None or parent_indices.numel() == 0:
        return zero, zero

    parent_points = label_points.index_select(0, parent_indices)
    child_points = label_points.index_select(0, child_indices)
    violation, angle, aperture = exact_entailment_cone_violation(
        parent_points,
        child_points,
        curv=curv,
        min_radius=min_radius,
        margin=margin,
        eps=eps,
    )
    raw_margin = angle - aperture + margin

    if hard_pool_ratio is not None and 0 < hard_pool_ratio < 1.0 and raw_margin.numel() > 1:
        hard_count = max(1, int(math.ceil(raw_margin.numel() * float(hard_pool_ratio))))
        raw_margin = torch.topk(raw_margin, k=hard_count, largest=True).values

    smooth_band = max(float(smooth_band), 1e-4)
    cone_loss = F.softplus(raw_margin / smooth_band).mean() * smooth_band
    violation_ratio = (violation > 0).to(label_points.dtype).mean()
    return cone_loss, violation_ratio


def build_depth_positive_targets(flat_labels, depth_label_indices):
    num_depths = len(depth_label_indices)
    batch_size = flat_labels.size(0)
    labels_by_depth = flat_labels.view(batch_size, num_depths, -1)
    label_level_mask = torch.zeros((batch_size, num_depths), device=flat_labels.device, dtype=torch.bool)
    positive_count_by_depth = labels_by_depth.new_zeros((batch_size, num_depths), dtype=torch.float32)
    positive_mask_by_depth = []

    for depth_idx, depth_label_ids in enumerate(depth_label_indices):
        if depth_label_ids is None or depth_label_ids.numel() == 0:
            positive_mask_by_depth.append(None)
            continue

        depth_label_ids = depth_label_ids.to(device=flat_labels.device, dtype=torch.long)
        depth_targets = labels_by_depth[:, depth_idx].index_select(1, depth_label_ids.to(flat_labels.device))
        positive_mask = depth_targets == 1
        positive_mask_by_depth.append(positive_mask)
        valid_samples = positive_mask.any(dim=1)
        if not valid_samples.any():
            continue

        label_level_mask[valid_samples, depth_idx] = True
        positive_count_by_depth[:, depth_idx] = positive_mask.sum(dim=1).to(dtype=torch.float32)

    return {
        'valid_depth_mask': label_level_mask,
        'positive_count_by_depth': positive_count_by_depth,
        'positive_mask_by_depth': positive_mask_by_depth,
    }


def _adjacent_edge_indices(valid_depth_mask):
    edge_mask = valid_depth_mask[:, :-1] & valid_depth_mask[:, 1:]
    if not edge_mask.any():
        return None, None
    batch_indices, depth_indices = torch.nonzero(edge_mask, as_tuple=True)
    return batch_indices, depth_indices


def _build_global_positive_mask(positive_mask_by_depth, depth_label_indices, batch_size, num_labels, device):
    positive_global = torch.zeros((batch_size, num_labels), device=device, dtype=torch.bool)
    for depth_idx, depth_label_ids in enumerate(depth_label_indices):
        if depth_label_ids is None or depth_label_ids.numel() == 0:
            continue
        positive_mask = positive_mask_by_depth[depth_idx]
        if positive_mask is None:
            continue
        positive_global[:, depth_label_ids.to(device=device, dtype=torch.long)] = positive_mask.to(device=device)
    return positive_global


def _extract_sample_endpoint_paths(sample_positive_mask, label_parent_indices):
    positive_labels = torch.nonzero(sample_positive_mask, as_tuple=False).flatten()
    if positive_labels.numel() == 0:
        return []

    positive_label_list = [int(label.item()) for label in positive_labels]
    endpoints = []
    for label_id in positive_label_list:
        has_positive_child = False
        for child_id in positive_label_list:
            if int(label_parent_indices[child_id].item()) == label_id:
                has_positive_child = True
                break
        if not has_positive_child:
            endpoints.append(label_id)

    if not endpoints:
        endpoints = positive_label_list

    paths = []
    seen = set()
    for endpoint in endpoints:
        path = []
        current = endpoint
        visited = set()
        while current != -1 and current not in visited:
            visited.add(current)
            path.append(current)
            current = int(label_parent_indices[current].item())
        path = tuple(reversed(path))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def query_to_label_tree_alignment_loss(
    query_tangent_by_depth,
    query_points_by_depth,
    label_points,
    flat_labels,
    depth_label_indices,
    label_parent_indices,
    label_depth_ids,
    depth_curvatures,
    margin=0.0,
    min_radius=0.1,
    eps=1e-6,
    softmin_temperature=1.0,
):
    zero = query_tangent_by_depth.new_zeros(())
    # L_q2path should not drag the structural prototype branch directly. The head
    # already builds alignment label points with a detached structural component
    # plus a live semantic component, so we keep gradients here to couple query
    # alignment with the label semantics used by classification.
    label_points_for_alignment = label_points
    depth_targets = build_depth_positive_targets(
        flat_labels=flat_labels,
        depth_label_indices=depth_label_indices,
    )
    valid_depth_mask = depth_targets['valid_depth_mask']
    positive_count_by_depth = depth_targets['positive_count_by_depth']
    positive_mask_by_depth = depth_targets['positive_mask_by_depth']
    positive_global_mask = _build_global_positive_mask(
        positive_mask_by_depth=positive_mask_by_depth,
        depth_label_indices=depth_label_indices,
        batch_size=query_tangent_by_depth.size(0),
        num_labels=label_points.size(0),
        device=label_points.device,
    )

    aux = {
        'valid_depth_mask': valid_depth_mask,
        'query_points_by_depth': query_points_by_depth,
        'alignment_loss': zero,
        'query_chain_loss': zero,
        'query_label_align_dist_mean': zero,
        'query_label_min_dist_mean': zero,
        'query_chain_violation_mean': zero,
        'valid_depth_ratio': valid_depth_mask.to(query_tangent_by_depth.dtype).mean(),
        'positive_count_by_depth': positive_count_by_depth,
        'positive_count_mean': positive_count_by_depth[valid_depth_mask].mean()
        if valid_depth_mask.any()
        else zero,
        'path_count_mean': zero,
        'path_length_mean': zero,
        'alignment_mean_by_depth': query_tangent_by_depth.new_zeros((query_tangent_by_depth.size(1),)),
        'min_distance_mean_by_depth': query_tangent_by_depth.new_zeros((query_tangent_by_depth.size(1),)),
        'valid_edge_mask': valid_depth_mask[:, :-1] & valid_depth_mask[:, 1:],
    }

    path_alignment_values = []
    min_path_distance_values = []
    min_distance_values = []
    alignment_mean_by_depth = query_tangent_by_depth.new_zeros((query_tangent_by_depth.size(1),))
    min_distance_mean_by_depth = query_tangent_by_depth.new_zeros((query_tangent_by_depth.size(1),))
    safe_temperature = max(float(softmin_temperature), 1e-4)
    path_counts = []
    path_lengths = []

    for depth_idx, depth_label_ids in enumerate(depth_label_indices):
        if depth_label_ids is None or depth_label_ids.numel() == 0:
            continue

        positive_mask = positive_mask_by_depth[depth_idx]
        if positive_mask is None:
            continue

        valid_samples = valid_depth_mask[:, depth_idx]
        if not valid_samples.any():
            continue

        depth_label_ids = depth_label_ids.to(device=label_points.device, dtype=torch.long)
        depth_label_points = label_points_for_alignment.index_select(0, depth_label_ids)
        query_depth_points = query_points_by_depth[valid_samples, depth_idx, :]
        depth_distances = pairwise_dist(
            query_depth_points,
            depth_label_points,
            curv=_resolve_depth_curvature(
                depth_curvatures,
                torch.full(
                    (query_depth_points.size(0),),
                    depth_idx,
                    device=query_depth_points.device,
                    dtype=torch.long,
                ),
                query_depth_points,
            )[0],
        )
        positive_mask_valid = positive_mask[valid_samples].to(device=depth_distances.device)
        positive_count_valid = positive_mask_valid.sum(dim=1).to(dtype=depth_distances.dtype).clamp_min(1.0)
        masked_logits = (-depth_distances / safe_temperature).masked_fill(
            ~positive_mask_valid,
            torch.finfo(depth_distances.dtype).min,
        )
        depth_softmin = -safe_temperature * (
            torch.logsumexp(masked_logits, dim=1) - torch.log(positive_count_valid)
        )
        positive_distances = depth_distances.masked_fill(
            ~positive_mask_valid,
            torch.finfo(depth_distances.dtype).max,
        )
        depth_min_distance = positive_distances.min(dim=1).values
        min_distance_values.append(depth_min_distance)
        alignment_mean_by_depth[depth_idx] = depth_softmin.mean().detach()
        min_distance_mean_by_depth[depth_idx] = depth_min_distance.mean().detach()

    for sample_idx in range(query_points_by_depth.size(0)):
        sample_paths = _extract_sample_endpoint_paths(
            positive_global_mask[sample_idx],
            label_parent_indices=label_parent_indices,
        )
        if not sample_paths:
            continue

        sample_path_energies = []
        for path in sample_paths:
            path_indices = torch.tensor(path, device=label_points.device, dtype=torch.long)
            path_depth_ids = label_depth_ids.index_select(0, path_indices)
            sample_query_path_points = query_points_by_depth[sample_idx].index_select(0, path_depth_ids)
            sample_label_path_points = label_points_for_alignment.index_select(0, path_indices)
            path_curvatures = _resolve_depth_curvature(depth_curvatures, path_depth_ids, sample_query_path_points)
            step_distances = lorentz_distance(
                sample_query_path_points,
                sample_label_path_points,
                curv=path_curvatures,
                eps=eps,
            )
            # Put more emphasis on fine-grained nodes so the path loss cannot be
            # solved mainly by matching shallow shared ancestors.
            step_weights = torch.arange(
                1,
                step_distances.size(0) + 1,
                device=step_distances.device,
                dtype=step_distances.dtype,
            )
            step_weights = step_weights / step_weights.sum().clamp_min(1e-6)
            sample_path_energies.append(torch.sum(step_distances * step_weights))
            path_lengths.append(len(path))

        if not sample_path_energies:
            continue

        path_energy_tensor = torch.stack(sample_path_energies)
        path_counts.append(len(sample_path_energies))
        softmin_path_energy = -safe_temperature * (
            torch.logsumexp(-path_energy_tensor / safe_temperature, dim=0) - math.log(len(sample_path_energies))
        )
        path_alignment_values.append(softmin_path_energy)
        min_path_distance_values.append(path_energy_tensor.min())

    if path_alignment_values:
        alignment_tensor = torch.stack(path_alignment_values)
        min_path_tensor = torch.stack(min_path_distance_values)
        alignment_loss = alignment_tensor.mean()
        aux['alignment_loss'] = alignment_loss.detach()
        aux['query_label_align_dist_mean'] = alignment_tensor.mean().detach()
        aux['query_label_min_dist_mean'] = min_path_tensor.mean().detach()
        aux['path_count_mean'] = alignment_tensor.new_tensor(sum(path_counts) / len(path_counts))
        aux['path_length_mean'] = alignment_tensor.new_tensor(sum(path_lengths) / len(path_lengths))
    else:
        alignment_loss = zero

    if min_distance_values:
        min_distance_tensor = torch.cat(min_distance_values, dim=0)
        aux['min_distance_mean_by_depth'] = min_distance_mean_by_depth
        aux['query_positive_min_dist_mean'] = min_distance_tensor.mean().detach()

    batch_indices, depth_indices = _adjacent_edge_indices(valid_depth_mask)
    if batch_indices is not None:
        # Keep the query-chain cone only as a diagnostic. L_q2path should stay a pure
        # query-to-gold-path alignment term, while query monotonicity is handled by
        # the radius parameterization inside the hyperbolic head.
        with torch.no_grad():
            parent_curvatures = _resolve_depth_curvature(
                depth_curvatures,
                depth_indices,
                query_tangent_by_depth[batch_indices, depth_indices],
            )

            query_parent_points = exp_map0(
                query_tangent_by_depth[batch_indices, depth_indices],
                curv=parent_curvatures.unsqueeze(-1),
            )
            query_child_points = exp_map0(
                query_tangent_by_depth[batch_indices, depth_indices + 1],
                curv=parent_curvatures.unsqueeze(-1),
            )
            query_violation, _, _ = exact_entailment_cone_violation(
                query_parent_points,
                query_child_points,
                curv=parent_curvatures,
                min_radius=min_radius,
                margin=margin,
                eps=eps,
            )
            aux['query_chain_loss'] = query_violation.mean().detach()
            aux['query_chain_violation_mean'] = query_violation.mean().detach()

    aux['alignment_mean_by_depth'] = alignment_mean_by_depth
    if not path_alignment_values:
        return zero, aux

    return alignment_loss, aux
