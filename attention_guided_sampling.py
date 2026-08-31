import math
import types

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F

from generate import get_num_transfer_tokens


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _model_type(model):
    model = _unwrap_model(model)
    return str(getattr(getattr(model, "config", None), "model_type", "")).lower()


def _should_try_output_attentions(model, attention_capture_backend):
    if attention_capture_backend == "patch_last_layer":
        return False
    if attention_capture_backend == "output_attentions":
        return True
    if attention_capture_backend != "auto":
        raise ValueError(f"Unknown attention_capture_backend: {attention_capture_backend!r}")

    # Dream already uses the patched capture path. LLaDA advertises no
    # output_attentions support, and probing it every denoising step adds
    # avoidable overhead on the hot path.
    model_type = _model_type(model)
    return model_type not in {"dream", "llada"}


def _get_last_block(model):
    model = _unwrap_model(model)
    transformer = getattr(model.model, "transformer", model.model)
    if hasattr(transformer, "blocks"):
        return transformer.blocks[-1]
    if hasattr(transformer, "block_groups"):
        return transformer.block_groups[-1][-1]
    if hasattr(transformer, "layers"):
        return transformer.layers[-1]
    raise AttributeError("Unable to locate the last transformer block for attention capture.")


def _iter_transformer_blocks(transformer):
    if hasattr(transformer, "blocks"):
        for block in transformer.blocks:
            yield block
        return
    if hasattr(transformer, "block_groups"):
        for group in transformer.block_groups:
            if isinstance(group, (list, tuple)):
                for block in group:
                    yield block
            else:
                yield group
        return
    if hasattr(transformer, "layers"):
        for block in transformer.layers:
            yield block
        return
    raise AttributeError("Unable to locate transformer blocks for attention capture.")


def _reduce_attention_heads(layer_attention, reduction):
    if reduction == "mean":
        return layer_attention.mean(dim=1)
    if reduction == "max":
        return layer_attention.max(dim=1).values
    if reduction == "entropy_weighted":
        probs = torch.clamp(layer_attention, min=0.0)
        entropy = -(probs * torch.log(torch.clamp(probs, min=torch.finfo(probs.dtype).eps))).sum(dim=-1).mean(dim=-1)
        weights = torch.softmax(-entropy, dim=-1).view(probs.shape[0], probs.shape[1], 1, 1)
        return (layer_attention * weights).sum(dim=1)
    raise ValueError(f"Unknown attention head reduction: {reduction}")


def _head_reduction_for_aggregation(aggregation):
    if aggregation in {
        "last_layer_mean",
        "first_layer_mean",
        "last_k_layers_mean",
        "first_k_layers_mean",
        "all_layers_mean",
    }:
        return "mean"
    if aggregation == "last_layer_max":
        return "max"
    if aggregation == "last_layer_entropy_weighted":
        return "entropy_weighted"
    raise ValueError(f"Unknown attention capture aggregation: {aggregation}")


def _capture_layer_indices(num_layers, aggregation="last_layer_mean", last_k_layers=2):
    if num_layers <= 0:
        raise ValueError("No transformer layers are available for attention capture.")

    if aggregation in {"last_layer_mean", "last_layer_max", "last_layer_entropy_weighted"}:
        return [num_layers - 1]
    if aggregation == "first_layer_mean":
        return [0]
    if aggregation == "last_k_layers_mean":
        k = max(1, min(int(last_k_layers), num_layers))
        return list(range(num_layers - k, num_layers))
    if aggregation == "first_k_layers_mean":
        k = max(1, min(int(last_k_layers), num_layers))
        return list(range(k))
    if aggregation == "all_layers_mean":
        return list(range(num_layers))
    raise ValueError(f"Unknown attention capture aggregation: {aggregation}")


def _aggregate_reduced_attention_layers(reduced_layers, aggregation="last_layer_mean"):
    if not reduced_layers:
        raise ValueError("No reduced attention layers were captured.")

    if aggregation in {
        "last_layer_mean",
        "first_layer_mean",
        "last_layer_max",
        "last_layer_entropy_weighted",
    }:
        return reduced_layers[0]

    if aggregation in {"last_k_layers_mean", "first_k_layers_mean", "all_layers_mean"}:
        return torch.stack(reduced_layers, dim=0).mean(dim=0)

    raise ValueError(f"Unknown attention capture aggregation: {aggregation}")


def _aggregate_attention_layers(layer_attentions, aggregation="last_layer_mean", last_k_layers=2):
    if not layer_attentions:
        raise ValueError("No attention layers were captured.")

    reduction = _head_reduction_for_aggregation(aggregation)
    indices = _capture_layer_indices(len(layer_attentions), aggregation=aggregation, last_k_layers=last_k_layers)
    reduced_layers = [_reduce_attention_heads(layer_attentions[idx], reduction) for idx in indices]
    return _aggregate_reduced_attention_layers(reduced_layers, aggregation=aggregation)


def _prepare_forward_attention_inputs(model, input_ids, attention_mask):
    model = _unwrap_model(model)
    model_type = _model_type(model)
    if model_type != "dream":
        return {"attention_mask": attention_mask}

    if attention_mask is None:
        return {"attention_mask": None, "position_ids": None}

    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        if torch.any(attention_mask == 0):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            allowed = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2).bool(),
                attention_mask.unsqueeze(1).unsqueeze(-1).bool(),
            )
            model_dtype = None
            try:
                model_dtype = next(model.parameters()).dtype
            except (StopIteration, AttributeError, TypeError):
                model_dtype = torch.float32
            additive_mask = torch.zeros_like(allowed, dtype=model_dtype)
            additive_mask = additive_mask.masked_fill(~allowed, torch.finfo(additive_mask.dtype).min)
            return {"attention_mask": additive_mask, "position_ids": position_ids}
        return {"attention_mask": None, "position_ids": None}

    return {"attention_mask": attention_mask}


def _align_logits_for_generation(model, logits):
    """Apply model-specific logit alignment used by native generation helpers."""
    if _model_type(model) == "dream":
        # Dream's native diffusion_generate shifts logits by one position before
        # sampling, so position i consumes the distribution produced for i-1.
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def _top_p_logits(logits, top_p=None):
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits, top_k=None):
    if top_k is None:
        return logits
    top_k = min(int(top_k), logits.size(-1))
    if top_k <= 0:
        return logits
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def _normalize_base_alg(base_alg):
    if base_alg in {"confidence", "maskgit_plus", None}:
        return "maskgit_plus"
    if base_alg in {"margin", "topk_margin"}:
        return "topk_margin"
    if base_alg == "entropy":
        return "entropy"
    raise ValueError(f"Unknown attention discount base algorithm: {base_alg}")


def _sample_tokens_and_scores(
    logits,
    temperature,
    top_p=None,
    top_k=None,
    base_alg="maskgit_plus",
    uncertainty_from_filtered_probs=True,
):
    base_alg = _normalize_base_alg(base_alg)
    entropy_logits = logits
    if temperature > 0:
        entropy_logits = entropy_logits / temperature
    sampling_logits = entropy_logits
    sampling_logits = _top_p_logits(sampling_logits, top_p=top_p)
    sampling_logits = _top_k_logits(sampling_logits, top_k=top_k)
    probs = torch.softmax(sampling_logits.float(), dim=-1)
    uncertainty_probs = probs if uncertainty_from_filtered_probs else torch.softmax(entropy_logits.float(), dim=-1)

    if temperature > 0:
        # Mirror Dream's native generation helper: if filtered probabilities
        # become invalid for categorical sampling, fall back to greedy decoding
        # instead of failing the whole run.
        try:
            flat_probs = probs.reshape(-1, probs.shape[-1])
            x0 = dists.Categorical(probs=flat_probs).sample().view(logits.shape[:-1])
            token_confidence = torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
        except Exception:
            token_confidence, x0 = probs.max(dim=-1)
    else:
        token_confidence, x0 = probs.max(dim=-1)

    uncertainty_token_confidence = torch.gather(uncertainty_probs, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)

    eps = torch.finfo(uncertainty_probs.dtype).eps
    entropy = -torch.sum(uncertainty_probs * torch.log(torch.clamp(uncertainty_probs, min=eps)), dim=-1)
    support = torch.count_nonzero(uncertainty_probs > 0, dim=-1)
    max_entropy = torch.log(torch.clamp(support, min=2).to(uncertainty_probs.dtype))
    normalized_entropy = torch.where(
        support > 1,
        entropy / torch.clamp(max_entropy, min=eps),
        torch.zeros_like(entropy),
    )

    if base_alg == "maskgit_plus":
        score = token_confidence
    elif base_alg == "topk_margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        score = sorted_probs[..., 0] - sorted_probs[..., 1]
    elif base_alg == "entropy":
        score = torch.sum(uncertainty_probs * torch.log(torch.clamp(uncertainty_probs, min=eps)), dim=-1)
    else:
        raise ValueError(f"Unknown attention discount base algorithm: {base_alg}")

    return x0, score, token_confidence, entropy, normalized_entropy, uncertainty_token_confidence


def _forward_with_last_layer_attention(
    model,
    input_ids,
    attention_mask=None,
    attention_capture_aggregation="last_layer_mean",
    attention_capture_last_k_layers=2,
    attention_capture_backend="auto",
):
    model = _unwrap_model(model)
    forward_kwargs = _prepare_forward_attention_inputs(model, input_ids, attention_mask)
    if _should_try_output_attentions(model, attention_capture_backend):
        try:
            outputs = model(
                input_ids,
                output_attentions=True,
                return_dict=True,
                **forward_kwargs,
            )
            if getattr(outputs, "attentions", None):
                layer_attentions = [layer for layer in outputs.attentions if layer is not None]
                if layer_attentions:
                    return (
                        outputs.logits,
                        _aggregate_attention_layers(
                            layer_attentions,
                            aggregation=attention_capture_aggregation,
                            last_k_layers=attention_capture_last_k_layers,
                        ),
                    )
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            pass

    transformer = getattr(model.model, "transformer", model.model)
    blocks = list(_iter_transformer_blocks(transformer))
    capture_indices = _capture_layer_indices(
        len(blocks),
        aggregation=attention_capture_aggregation,
        last_k_layers=attention_capture_last_k_layers,
    )
    capture_index_to_slot = {layer_idx: slot_idx for slot_idx, layer_idx in enumerate(capture_indices)}
    reduction = _head_reduction_for_aggregation(attention_capture_aggregation)
    originals = []
    captured_layers = [None] * len(capture_indices)

    def _dream_forward_with_captured_attention(
        self_attn,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        cache_position=None,
        position_embeddings=None,
    ):
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        bsz, q_len, _ = hidden_states.size()
        query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

        forward_globals = self_attn.__class__.forward.__globals__
        apply_rotary_pos_emb = forward_globals["apply_rotary_pos_emb"]
        repeat_kv = forward_globals["repeat_kv"]

        if position_embeddings is None:
            cos, sin = self_attn.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self_attn.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self_attn.num_key_value_groups)

        compute_dtype = query_states.dtype
        scores = torch.matmul(query_states.to(compute_dtype), key_states.to(compute_dtype).transpose(2, 3)) / math.sqrt(self_attn.head_dim)
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            scores = scores + attention_mask[:, :, :, : key_states.shape[-2]].to(compute_dtype)
        # Match DreamAttention.forward: compute the attention distribution with an
        # fp32 softmax before casting back down. This keeps the captured
        # last-layer attention numerically aligned with Dream's manual
        # output_attentions path.
        reduced_attention = _reduce_attention_heads(
            torch.softmax(scores, dim=-1, dtype=torch.float32).to(compute_dtype),
            reduction,
        ).detach()

        if query_states.device.type == "cuda" and attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_mask_tensor = None
        if isinstance(attention_mask, torch.Tensor):
            attn_mask_tensor = attention_mask.to(query_states.dtype)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask_tensor,
            dropout_p=self_attn.attention_dropout if self_attn.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self_attn.hidden_size)
        attn_output = self_attn.o_proj(attn_output)

        return attn_output, reduced_attention, past_key_value

    def make_patched(slot_idx, original):
        def patched(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            q_local = q
            k_local = k
            v_local = v
            assert k_local.size(1) == v_local.size(1)
            if q_local.size(1) != k_local.size(1):
                repeat = q_local.size(1) // k_local.size(1)
                k_local = k_local.repeat_interleave(repeat, dim=1, output_size=q_local.size(1))
                v_local = v_local.repeat_interleave(repeat, dim=1, output_size=q_local.size(1))

            compute_dtype = q_local.dtype
            scores = torch.matmul(q_local.to(compute_dtype), k_local.to(compute_dtype).transpose(-1, -2)) / math.sqrt(q_local.size(-1))
            if attn_mask is not None:
                scores = scores + attn_mask.to(compute_dtype)
            captured_layers[slot_idx] = _reduce_attention_heads(torch.softmax(scores, dim=-1), reduction).detach()
            return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        return patched

    for layer_idx in capture_indices:
        block = blocks[layer_idx]
        slot_idx = capture_index_to_slot[layer_idx]
        if hasattr(block, "_scaled_dot_product_attention"):
            original = block._scaled_dot_product_attention
            originals.append(("sdpa", block, original))
            block._scaled_dot_product_attention = types.MethodType(make_patched(slot_idx, original), block)
        elif hasattr(block, "self_attn") and hasattr(block.self_attn, "forward"):
            original = block.self_attn.forward

            def make_dream_patched(slot_idx, original):
                def patched(
                    self,
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=None,
                ):
                    if output_attentions:
                        return original(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                    attn_output, reduced_attention, new_past_key_value = _dream_forward_with_captured_attention(
                        self,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    captured_layers[slot_idx] = reduced_attention
                    return attn_output, None, new_past_key_value

                return patched

            originals.append(("dream_forward", block.self_attn, original))
            block.self_attn.forward = types.MethodType(make_dream_patched(slot_idx, original), block.self_attn)
        else:
            raise RuntimeError("Unable to patch attention capture for the selected transformer block.")

    try:
        outputs = model(input_ids, return_dict=True, **forward_kwargs)
    finally:
        for patch_type, module, original in originals:
            if patch_type == "sdpa":
                module._scaled_dot_product_attention = original
            elif patch_type == "dream_forward":
                module.forward = original

    if any(layer_attention is None for layer_attention in captured_layers):
        raise RuntimeError("Failed to capture last-layer attention.")

    return (
        outputs.logits,
        _aggregate_reduced_attention_layers(
            captured_layers,
            aggregation=attention_capture_aggregation,
        ),
    )


def _masked_scope(masked_row, selected_index):
    scope = masked_row.clone()
    scope[selected_index] = True
    return scope


def _normalization_scope(normalization, masked_row, generation_scope, selected_index):
    if normalization == "masked_only":
        return _masked_scope(masked_row, selected_index)
    if normalization == "generated_span":
        return generation_scope
    raise ValueError(f"Unknown attention dependency normalization: {normalization}")


def _normalized_scores(row_or_col_scores, denom):
    eps = torch.finfo(row_or_col_scores.dtype).eps
    return row_or_col_scores / torch.clamp(denom, min=eps)


def _attention_dependency_scores_raw(attention_matrix, selected_index, mode):
    if mode == "mean_bidirectional":
        return 0.5 * (attention_matrix[selected_index] + attention_matrix[:, selected_index])
    if mode == "sum_bidirectional":
        return attention_matrix[selected_index] + attention_matrix[:, selected_index]
    if mode == "selected_to_candidate":
        return attention_matrix[selected_index]
    if mode == "candidate_to_selected":
        return attention_matrix[:, selected_index]
    raise ValueError(f"Unknown attention dependency mode: {mode}")


def _attention_dependency_scores_normalized(attention_matrix, selected_index, mode, scope):
    if mode == "mean_bidirectional":
        forward = attention_matrix[selected_index]
        backward = attention_matrix[:, selected_index]
        forward_norm = _normalized_scores(forward, forward[scope].sum())
        backward_norm = _normalized_scores(backward, attention_matrix[:, scope].sum(dim=-1))
        return 0.5 * (forward_norm + backward_norm)
    if mode == "sum_bidirectional":
        forward = attention_matrix[selected_index]
        backward = attention_matrix[:, selected_index]
        forward_norm = _normalized_scores(forward, forward[scope].sum())
        backward_norm = _normalized_scores(backward, attention_matrix[:, scope].sum(dim=-1))
        return forward_norm + backward_norm
    if mode == "selected_to_candidate":
        scores = attention_matrix[selected_index]
        return _normalized_scores(scores, scores[scope].sum())
    if mode == "candidate_to_selected":
        scores = attention_matrix[:, selected_index]
        return _normalized_scores(scores, attention_matrix[:, scope].sum(dim=-1))
    raise ValueError(f"Unknown attention dependency mode: {mode}")


def _attention_dependency_scores(
    attention_matrix,
    selected_index,
    mode,
    normalization,
    masked_row,
    generation_scope,
):
    if normalization == "raw":
        return _attention_dependency_scores_raw(attention_matrix, selected_index, mode)

    if normalization in {"masked_only", "generated_span"}:
        scope = _normalization_scope(normalization, masked_row, generation_scope, selected_index)
        return _attention_dependency_scores_normalized(attention_matrix, selected_index, mode, scope)

    if normalization == "hybrid_masked":
        raw_scores = _attention_dependency_scores_raw(attention_matrix, selected_index, mode)
        masked_scope = _masked_scope(masked_row, selected_index)
        masked_scores = _attention_dependency_scores_normalized(attention_matrix, selected_index, mode, masked_scope)
        return torch.sqrt(torch.clamp(raw_scores, min=0.0) * torch.clamp(masked_scores, min=0.0))

    raise ValueError(f"Unknown attention dependency normalization: {normalization}")


def _confidence_weighted_bidirectional_dependency(
    attention_matrix,
    selected_index,
    base_scores,
):
    candidate_confidence = torch.clamp(base_scores, min=0.0, max=1.0)
    selected_confidence = torch.clamp(base_scores[selected_index], min=0.0, max=1.0)
    return attention_matrix[selected_index] * (1.0 - candidate_confidence) + attention_matrix[:, selected_index] * (
        1.0 - selected_confidence
    )


def _apply_score_rule(base_scores, current_scores, dependency, alpha, score_rule, use_base_scores):
    if score_rule == "additive":
        return current_scores - alpha * dependency
    raise ValueError("Only additive attention discounting is supported in the paper implementation.")


def _selected_uncertainty_weight(confidence_value, mode):
    if mode == "none":
        return 1.0
    if mode == "confidence_gap":
        return float(1.0 - torch.clamp(confidence_value, min=0.0, max=1.0).item())
    raise ValueError("Only selected_uncertainty='confidence_gap' is used by the paper implementation.")


def _select_top_confidence(
    confidence_row,
    masked_row,
    num_select,
):
    selected = torch.zeros_like(masked_row)
    if int(num_select) <= 0:
        return selected
    scores = confidence_row.clone()
    scores[~masked_row] = -torch.inf
    valid = torch.count_nonzero(torch.isfinite(scores)).item()
    if valid <= 0:
        return selected
    k = min(int(num_select), int(valid))
    _, select_index = torch.topk(scores, k=k)
    selected[select_index] = True
    return selected


def _empty_selection_details(selected):
    return {
        "selected": selected,
        "selected_count": 0,
        "selected_risks": [],
        "budget_values": [],
        "accepted_confidences": [],
        "accepted_uncertainty_weights": [],
        "accepted_raw_dependencies": [],
        "selection_overlap_ratio": None,
        "selection_replacement_count": None,
        "selected_base_score_mean": None,
        "selected_effective_score_mean": None,
        "selected_score_shift_mean": None,
    }


def _selection_comparison_stats(selected, baseline_selected, selected_base_scores, selected_effective_scores):
    if not selected_base_scores:
        return {
            "selection_overlap_ratio": None,
            "selection_replacement_count": None,
            "selected_base_score_mean": None,
            "selected_effective_score_mean": None,
            "selected_score_shift_mean": None,
        }

    score_shifts = [base - eff for base, eff in zip(selected_base_scores, selected_effective_scores)]
    selected_count = int(selected.sum().item())
    overlap_count = int((selected & baseline_selected).sum().item())
    replacement_count = int((selected & ~baseline_selected).sum().item())
    return {
        "selection_overlap_ratio": float(overlap_count) / float(selected_count),
        "selection_replacement_count": replacement_count,
        "selected_base_score_mean": float(sum(selected_base_scores) / len(selected_base_scores)),
        "selected_effective_score_mean": float(sum(selected_effective_scores) / len(selected_effective_scores)),
        "selected_score_shift_mean": float(sum(score_shifts) / len(score_shifts)),
    }


def _select_with_attention_discount(
    confidence_row,
    selected_confidence_row,
    masked_row,
    attention_matrix,
    num_select,
    alpha,
    mode,
    score_rule,
    normalization,
    aggregation,
    selected_uncertainty,
    generation_scope,
    return_details=False,
):
    if score_rule != "additive":
        raise ValueError("Only additive attention discounting is supported in the paper implementation.")

    selected = torch.zeros_like(masked_row)
    if int(num_select) <= 0:
        return _empty_selection_details(selected) if return_details else selected

    available = masked_row.clone()
    probability_scores = confidence_row.clone()
    if score_rule == "logit_additive":
        base_scores = torch.logit(torch.clamp(probability_scores, min=1e-6, max=1.0 - 1e-6))
    else:
        base_scores = probability_scores.clone()
    scores = base_scores.clone()
    scores[~available] = -torch.inf

    aggregate_dependency = torch.zeros_like(scores)
    dependency_sum = torch.zeros_like(scores)
    dependency_count = 0

    baseline_selected = _select_top_confidence(
        confidence_row=confidence_row,
        masked_row=masked_row,
        num_select=num_select,
    )
    selected_base_scores = []
    selected_effective_scores = []
    selected_confidences = []
    selected_uncertainty_weights = []
    selected_raw_dependencies = []

    for _ in range(int(num_select)):
        selected_index = torch.argmax(scores)
        if not torch.isfinite(scores[selected_index]):
            break
        selected_base_scores.append(float(base_scores[selected_index].item()))
        selected_effective_scores.append(float(scores[selected_index].item()))

        selected[selected_index] = True
        available[selected_index] = False
        scores[selected_index] = -torch.inf

        if mode == "confidence_weighted_bidirectional":
            if normalization != "raw":
                raise ValueError("confidence_weighted_bidirectional currently supports normalization='raw' only.")
            dependency = _confidence_weighted_bidirectional_dependency(
                attention_matrix,
                int(selected_index.item()),
                probability_scores,
            )
        else:
            dependency = _attention_dependency_scores(
                attention_matrix,
                int(selected_index.item()),
                mode,
                normalization,
                available,
                generation_scope,
            )

        selected_confidence = (
            selected_confidence_row[selected_index]
            if selected_confidence_row is not None
            else probability_scores[selected_index]
        )
        uncertainty_weight = _selected_uncertainty_weight(selected_confidence, selected_uncertainty)
        dependency = dependency * uncertainty_weight

        if aggregation == "sequential":
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=(score_rule == "mmr"),
            )
        elif aggregation == "max":
            aggregate_dependency = torch.maximum(aggregate_dependency, dependency)
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "mean":
            dependency_sum = dependency_sum + dependency
            dependency_count += 1
            aggregate_dependency = dependency_sum / dependency_count
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "sum":
            dependency_sum = dependency_sum + dependency
            aggregate_dependency = dependency_sum
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        else:
            raise ValueError(f"Unknown attention discount aggregation: {aggregation}")

        scores = torch.where(available, next_scores, torch.full_like(scores, -torch.inf))
        selected_confidences.append(float(torch.clamp(selected_confidence, min=0.0, max=1.0).item()))
        selected_uncertainty_weights.append(float(uncertainty_weight))
        selected_raw_dependencies.append(float(torch.clamp(dependency[selected_index], min=0.0).item()))

    if not return_details:
        return selected

    details = _empty_selection_details(selected)
    details["selected_count"] = int(selected.sum().item())
    details["accepted_confidences"] = selected_confidences
    details["accepted_uncertainty_weights"] = selected_uncertainty_weights
    details["accepted_raw_dependencies"] = selected_raw_dependencies
    details.update(_selection_comparison_stats(selected, baseline_selected, selected_base_scores, selected_effective_scores))
    return details


def _select_with_attention_discount_ab_budget_eb_stop(
    confidence_row,
    selected_confidence_row,
    entropy_row,
    masked_row,
    attention_matrix,
    max_select,
    gamma,
    alpha,
    mode,
    score_rule,
    normalization,
    aggregation,
    selected_uncertainty,
    generation_scope,
    return_details=False,
):
    if score_rule != "additive":
        raise ValueError("Only additive attention discounting is supported in the paper implementation.")

    selected = torch.zeros_like(masked_row)
    available = masked_row.clone()
    probability_scores = confidence_row.clone()
    if score_rule == "logit_additive":
        base_scores = torch.logit(torch.clamp(probability_scores, min=1e-6, max=1.0 - 1e-6))
    else:
        base_scores = probability_scores.clone()
    scores = base_scores.clone()
    scores[~available] = -torch.inf

    aggregate_dependency = torch.zeros_like(scores)
    dependency_sum = torch.zeros_like(scores)
    dependency_count = 0

    target_cap = int(torch.count_nonzero(masked_row).item()) if max_select is None else max(1, int(max_select))
    selected_indices = []
    selected_mask = torch.zeros_like(masked_row, dtype=torch.bool)
    running_entropy_sum = 0.0
    running_entropy_max = 0.0
    running_raw_budget = 0.0
    budget_values = []
    selected_risks = []
    accepted_confidences = []
    accepted_uncertainty_weights = []
    accepted_raw_dependencies = []
    selected_base_scores = []
    selected_effective_scores = []

    while len(selected_indices) < target_cap:
        selected_index = torch.argmax(scores)
        if not torch.isfinite(scores[selected_index]):
            break

        candidate_index = int(selected_index.item())
        candidate_entropy = float(torch.clamp(entropy_row[selected_index], min=0.0).item())
        budget_value = (running_entropy_sum + candidate_entropy) - max(running_entropy_max, candidate_entropy)

        if selected_indices:
            raw_increment = 2.0 * float(
                attention_matrix[candidate_index, selected_mask].to(torch.float32).sum().item()
                + attention_matrix[selected_mask, candidate_index].to(torch.float32).sum().item()
            )
        else:
            raw_increment = 0.0
        raw_dependency_value = running_raw_budget + raw_increment

        if selected_indices and budget_value > float(gamma):
            break

        selected_base_scores.append(float(base_scores[selected_index].item()))
        selected_effective_scores.append(float(scores[selected_index].item()))
        selected[selected_index] = True
        available[selected_index] = False
        scores[selected_index] = -torch.inf
        selected_indices.append(candidate_index)
        selected_mask[selected_index] = True
        running_entropy_sum += candidate_entropy
        running_entropy_max = max(running_entropy_max, candidate_entropy)
        running_raw_budget = raw_dependency_value

        budget_values.append(float(budget_value))
        selected_risks.append(candidate_entropy)
        accepted_raw_dependencies.append(float(raw_increment))

        selected_confidence = (
            selected_confidence_row[selected_index]
            if selected_confidence_row is not None
            else probability_scores[selected_index]
        )
        confidence_value = float(torch.clamp(selected_confidence, min=0.0, max=1.0).item())
        accepted_confidences.append(confidence_value)
        accepted_uncertainty_weights.append(float(1.0 - confidence_value))

        if mode == "confidence_weighted_bidirectional":
            if normalization != "raw":
                raise ValueError("confidence_weighted_bidirectional currently supports normalization='raw' only.")
            dependency = _confidence_weighted_bidirectional_dependency(
                attention_matrix,
                candidate_index,
                probability_scores,
            )
        else:
            dependency = _attention_dependency_scores(
                attention_matrix,
                candidate_index,
                mode,
                normalization,
                available,
                generation_scope,
            )

        uncertainty_weight = _selected_uncertainty_weight(selected_confidence, selected_uncertainty)
        dependency = dependency * uncertainty_weight

        if aggregation == "sequential":
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=(score_rule == "mmr"),
            )
        elif aggregation == "max":
            aggregate_dependency = torch.maximum(aggregate_dependency, dependency)
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "mean":
            dependency_sum = dependency_sum + dependency
            dependency_count += 1
            aggregate_dependency = dependency_sum / dependency_count
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "sum":
            dependency_sum = dependency_sum + dependency
            aggregate_dependency = dependency_sum
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        else:
            raise ValueError(f"Unknown attention discount aggregation: {aggregation}")

        scores = torch.where(available, next_scores, torch.full_like(scores, -torch.inf))

    if not return_details:
        return selected

    baseline_selected = _select_top_confidence(
        confidence_row=confidence_row,
        masked_row=masked_row,
        num_select=int(selected.sum().item()),
    )
    details = _empty_selection_details(selected)
    details["selected_count"] = int(selected.sum().item())
    details["selected_risks"] = selected_risks
    details["budget_values"] = budget_values
    details["accepted_confidences"] = accepted_confidences
    details["accepted_uncertainty_weights"] = accepted_uncertainty_weights
    details["accepted_raw_dependencies"] = accepted_raw_dependencies
    details.update(_selection_comparison_stats(selected, baseline_selected, selected_base_scores, selected_effective_scores))
    return details


def _select_with_fast_dllm(
    confidence_row,
    masked_row,
    gamma,
    return_details=False,
):
    selected = torch.zeros_like(masked_row)
    scores = confidence_row.clone()
    scores[~masked_row] = -torch.inf
    valid_count = int(torch.count_nonzero(torch.isfinite(scores)).item())
    if valid_count <= 0:
        return _empty_selection_details(selected) if return_details else selected

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    sorted_scores = sorted_scores[:valid_count]
    sorted_indices = sorted_indices[:valid_count]

    ranks = torch.arange(1, valid_count + 1, device=scores.device, dtype=torch.float32)
    confidences = torch.clamp(sorted_scores.to(torch.float32), min=0.0, max=1.0)
    budget_values_tensor = (ranks + 1.0) * (1.0 - confidences)
    feasible = budget_values_tensor < float(gamma)
    if bool(torch.any(feasible)):
        num_select = int(torch.nonzero(feasible, as_tuple=False)[-1].item()) + 1
    else:
        num_select = 1

    selected_indices = sorted_indices[:num_select]
    selected[selected_indices] = True

    if not return_details:
        return selected

    selected_confidences = [float(v) for v in confidences[:num_select].detach().cpu().tolist()]
    budget_values = [float(v) for v in budget_values_tensor[:num_select].detach().cpu().tolist()]
    selected_risks = [float(1.0 - v) for v in selected_confidences]
    details = _empty_selection_details(selected)
    details["selected_count"] = int(num_select)
    details["selected_risks"] = selected_risks
    details["budget_values"] = budget_values
    details["accepted_confidences"] = selected_confidences
    details["accepted_uncertainty_weights"] = selected_risks
    details["accepted_raw_dependencies"] = []
    details.update(
        _selection_comparison_stats(
            selected,
            selected.clone(),
            selected_confidences,
            selected_confidences,
        )
    )
    return details


def _select_with_attention_discount_fast_dllm(
    confidence_row,
    selected_confidence_row,
    masked_row,
    attention_matrix,
    max_select,
    gamma,
    alpha,
    mode,
    score_rule,
    normalization,
    aggregation,
    selected_uncertainty,
    generation_scope,
    return_details=False,
):
    if score_rule != "additive":
        raise ValueError("Only additive attention discounting is supported in the paper implementation.")

    selected = torch.zeros_like(masked_row)
    available = masked_row.clone()
    probability_scores = confidence_row.clone()
    if score_rule == "logit_additive":
        base_scores = torch.logit(torch.clamp(probability_scores, min=1e-6, max=1.0 - 1e-6))
    else:
        base_scores = probability_scores.clone()
    scores = base_scores.clone()
    scores[~available] = -torch.inf

    aggregate_dependency = torch.zeros_like(scores)
    dependency_sum = torch.zeros_like(scores)
    dependency_count = 0

    target_cap = int(torch.count_nonzero(masked_row).item()) if max_select is None else max(1, int(max_select))
    selected_indices = []
    min_confidence = 1.0
    budget_values = []
    selected_risks = []
    accepted_confidences = []
    accepted_uncertainty_weights = []
    accepted_raw_dependencies = []
    selected_base_scores = []
    selected_effective_scores = []

    while len(selected_indices) < target_cap:
        selected_index = torch.argmax(scores)
        if not torch.isfinite(scores[selected_index]):
            break

        candidate_index = int(selected_index.item())
        selected_confidence = (
            selected_confidence_row[selected_index]
            if selected_confidence_row is not None
            else probability_scores[selected_index]
        )
        confidence_value = float(torch.clamp(selected_confidence, min=0.0, max=1.0).item())
        candidate_min_confidence = min(min_confidence, confidence_value)
        candidate_count = len(selected_indices) + 1
        budget_value = (candidate_count + 1) * (1.0 - candidate_min_confidence)

        if selected_indices and budget_value >= float(gamma):
            break

        selected_base_scores.append(float(base_scores[selected_index].item()))
        selected_effective_scores.append(float(scores[selected_index].item()))
        selected[selected_index] = True
        available[selected_index] = False
        scores[selected_index] = -torch.inf
        selected_indices.append(candidate_index)
        min_confidence = candidate_min_confidence

        budget_values.append(float(budget_value))
        selected_risks.append(float(1.0 - confidence_value))
        accepted_confidences.append(confidence_value)
        accepted_uncertainty_weights.append(float(1.0 - confidence_value))
        accepted_raw_dependencies.append(0.0)

        if mode == "confidence_weighted_bidirectional":
            if normalization != "raw":
                raise ValueError("confidence_weighted_bidirectional currently supports normalization='raw' only.")
            dependency = _confidence_weighted_bidirectional_dependency(
                attention_matrix,
                candidate_index,
                probability_scores,
            )
        else:
            dependency = _attention_dependency_scores(
                attention_matrix,
                candidate_index,
                mode,
                normalization,
                available,
                generation_scope,
            )

        uncertainty_weight = _selected_uncertainty_weight(selected_confidence, selected_uncertainty)
        dependency = dependency * uncertainty_weight

        if aggregation == "sequential":
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=(score_rule == "mmr"),
            )
        elif aggregation == "max":
            aggregate_dependency = torch.maximum(aggregate_dependency, dependency)
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "mean":
            dependency_sum = dependency_sum + dependency
            dependency_count += 1
            aggregate_dependency = dependency_sum / dependency_count
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        elif aggregation == "sum":
            dependency_sum = dependency_sum + dependency
            aggregate_dependency = dependency_sum
            next_scores = _apply_score_rule(
                base_scores=base_scores,
                current_scores=scores,
                dependency=aggregate_dependency,
                alpha=alpha,
                score_rule=score_rule,
                use_base_scores=True,
            )
        else:
            raise ValueError(f"Unknown attention discount aggregation: {aggregation}")

        scores = torch.where(available, next_scores, torch.full_like(scores, -torch.inf))

    if not return_details:
        return selected

    baseline_selected = _select_with_fast_dllm(
        confidence_row=confidence_row,
        masked_row=masked_row,
        gamma=gamma,
        return_details=False,
    )
    details = _empty_selection_details(selected)
    details["selected_count"] = int(selected.sum().item())
    details["selected_risks"] = selected_risks
    details["budget_values"] = budget_values
    details["accepted_confidences"] = accepted_confidences
    details["accepted_uncertainty_weights"] = accepted_uncertainty_weights
    details["accepted_raw_dependencies"] = accepted_raw_dependencies
    details.update(_selection_comparison_stats(selected, baseline_selected, selected_base_scores, selected_effective_scores))
    return details


def _select_with_eb_paper_exact(
    error_proxy_row,
    entropy_row,
    masked_row,
    gamma,
    return_details=False,
):
    available_indices = torch.nonzero(masked_row, as_tuple=False).flatten()
    if available_indices.numel() == 0:
        selected = torch.zeros_like(masked_row)
        if return_details:
            return {
                "selected": selected,
                "selected_risks": [],
                "budget_values": [],
                "accepted_confidences": [],
                "accepted_uncertainty_weights": [],
                "accepted_raw_dependencies": [],
            }
        return selected

    ranked_order = torch.argsort(error_proxy_row.index_select(0, available_indices), descending=False)
    ranked_indices = available_indices[ranked_order]
    ranked_entropy = entropy_row.index_select(0, ranked_indices).to(torch.float32)

    acc_entropy = torch.cumsum(ranked_entropy, dim=0)
    cummax_entropy = torch.cummax(ranked_entropy, dim=0).values
    prefix_budget = acc_entropy - cummax_entropy

    k = int(torch.count_nonzero(prefix_budget <= float(gamma)).item())
    k = min(k, int(available_indices.numel()))

    selected = torch.zeros_like(masked_row)
    if k > 0:
        selected[ranked_indices[:k]] = True

    if not return_details:
        return selected

    return {
        "selected": selected,
        "selected_risks": [float(v) for v in ranked_entropy[:k].tolist()],
        "budget_values": [float(v) for v in prefix_budget[:k].tolist()],
        "accepted_confidences": [None] * k,
        "accepted_uncertainty_weights": [None] * k,
        "accepted_raw_dependencies": [0.0] * k,
    }


@torch.no_grad()
def generate_attention_discounted(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    top_p=None,
    top_k=None,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    attention_discount_alpha=40.0,
    attention_discount_mode="candidate_to_selected",
    attention_discount_score_rule="additive",
    attention_discount_normalization="raw",
    attention_discount_aggregation="sequential",
    attention_discount_selected_uncertainty="confidence_gap",
    attention_discount_base_alg="maskgit_plus",
    attention_capture_aggregation="last_layer_mean",
    attention_capture_last_k_layers=2,
    attention_capture_backend="auto",
    sampler_variant="attention_discounted",
    dbes_gamma=1.0,
    abnn_f=1.0,
    pad_to_max_seq_len=False,
    max_seq_len=None,
    pad_token_id=0,
    return_stats=False,
):
    """Shared diffusion generation loop for confidence baselines and ADAS.

    Public sampler variants:
    - ``confidence_topk``: fixed-size confidence top-k baseline.
    - ``attention_discounted``: fixed-size top-k with attention-discounted ordering.
    - ``eb_entropy``: entropy-budget baseline.
    - ``attention_discount_ab_budget_eb_stop``: EB stopping with ADAS ordering.
    - ``fast_dllm``: Fast-DLLM stopping baseline.
    - ``ad_fast_dllm``: Fast-DLLM stopping with ADAS ordering.
    """
    if remasking != "low_confidence":
        raise ValueError("generate_attention_discounted currently supports remasking='low_confidence' only.")

    if sampler_variant == "eb_paper_exact":
        sampler_variant = "eb_entropy"

    supported_sampler_variants = {
        "attention_discounted",
        "confidence_topk",
        "eb_entropy",
        "attention_discount_ab_budget_eb_stop",
        "fast_dllm",
        "ad_fast_dllm",
    }
    if sampler_variant not in supported_sampler_variants:
        raise ValueError(f"Unknown sampler_variant: {sampler_variant}")

    if int(steps) <= 0 and sampler_variant in {"attention_discounted", "confidence_topk"}:
        raise ValueError(f"{sampler_variant} requires fixed positive steps.")

    model = _unwrap_model(model)
    prompt_len = prompt.shape[1]

    if pad_to_max_seq_len:
        if max_seq_len is None:
            max_seq_len = (
                getattr(getattr(model, "config", None), "max_position_embeddings", None)
                or getattr(getattr(model, "config", None), "max_seq_len", None)
                or getattr(getattr(model, "config", None), "n_positions", None)
            )
        if max_seq_len is None:
            raise ValueError("pad_to_max_seq_len=True requires max_seq_len or a model config max sequence length.")
        if max_seq_len < gen_length + 1:
            raise ValueError(f"max_seq_len={max_seq_len} is too small for gen_length={gen_length}.")
        max_prompt_len = max_seq_len - gen_length
        if prompt_len > max_prompt_len:
            prompt = prompt[:, -max_prompt_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_prompt_len:]
            prompt_len = prompt.shape[1]
        seq_len = max_seq_len
        x = torch.full((prompt.shape[0], seq_len), pad_token_id, dtype=torch.long, device=model.device)
        x[:, :prompt_len] = prompt.clone()
        x[:, prompt_len : prompt_len + gen_length] = mask_id
    else:
        seq_len = prompt_len + gen_length
        x = torch.full((prompt.shape[0], seq_len), mask_id, dtype=torch.long, device=model.device)
        x[:, :prompt_len] = prompt.clone()

    if attention_mask is not None and attention_mask.dtype not in (torch.bool, torch.float16, torch.float32, torch.float64, torch.bfloat16):
        attention_mask = attention_mask.to(torch.bool)

    if pad_to_max_seq_len:
        if attention_mask is None:
            attention_mask = torch.zeros((prompt.shape[0], seq_len), dtype=torch.bool, device=model.device)
            attention_mask[:, : prompt_len + gen_length] = True
        else:
            padded_attention_mask = torch.zeros((prompt.shape[0], seq_len), dtype=attention_mask.dtype, device=model.device)
            padded_attention_mask[:, :prompt_len] = attention_mask
            padded_attention_mask[:, prompt_len : prompt_len + gen_length] = True
            attention_mask = padded_attention_mask
    elif attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device),
            ],
            dim=-1,
        )

    prompt_index = torch.zeros_like(x, dtype=torch.bool, device=model.device)
    if attention_mask is not None:
        prompt_index[:, :prompt_len] = attention_mask[:, :prompt_len].to(torch.bool)
    else:
        prompt_index[:, :prompt_len] = True
    generation_scope = torch.zeros(x.shape[1], dtype=torch.bool, device=model.device)
    generation_scope[prompt_len : prompt_len + gen_length] = True
    nfe_per_sample = torch.zeros(prompt.shape[0], dtype=torch.int64, device=model.device)
    selected_counts_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_risks_per_sample = [[] for _ in range(prompt.shape[0])]
    step_budget_values_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_confidences_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_uncertainty_weights_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_raw_dependencies_per_sample = [[] for _ in range(prompt.shape[0])]
    selection_overlap_ratio_per_sample = [[] for _ in range(prompt.shape[0])]
    selection_replacement_count_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_base_score_mean_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_effective_score_mean_per_sample = [[] for _ in range(prompt.shape[0])]
    selected_score_shift_mean_per_sample = [[] for _ in range(prompt.shape[0])]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    if int(steps) > 0:
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
    else:
        steps_per_block = block_length

    for num_block in range(num_blocks):
        block_slice_start = prompt_len + num_block * block_length
        block_slice_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = x[:, block_slice_start:block_slice_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block) if int(steps) > 0 else None

        for step_idx in range(steps_per_block):
            active_block_mask = x[:, block_slice_start:block_slice_end] == mask_id
            active_samples = active_block_mask.any(dim=1)
            if not torch.any(active_samples):
                break
            nfe_per_sample[active_samples] += 1
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
            else:
                attention_mask_ = attention_mask

            needs_attention = sampler_variant in {
                "attention_discounted",
                "attention_discount_ab_budget_eb_stop",
                "ad_fast_dllm",
            }
            attention = None
            if needs_attention:
                logits, attention = _forward_with_last_layer_attention(
                    model,
                    x_ if cfg_scale > 0.0 else x,
                    attention_mask=attention_mask_,
                    attention_capture_aggregation=attention_capture_aggregation,
                    attention_capture_last_k_layers=attention_capture_last_k_layers,
                    attention_capture_backend=attention_capture_backend,
                )
                if cfg_scale > 0.0:
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    attention, _ = torch.chunk(attention, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                forward_kwargs = _prepare_forward_attention_inputs(
                    model,
                    x_ if cfg_scale > 0.0 else x,
                    attention_mask_,
                )
                logits = model(x_ if cfg_scale > 0.0 else x, return_dict=True, **forward_kwargs).logits
                if cfg_scale > 0.0:
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

            logits = _align_logits_for_generation(model, logits)

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            score_logits = logits
            if confidence_eos_eot_inf:
                score_logits = logits.clone()
                score_logits[:, :, 126081] = -torch.inf
                score_logits[:, :, 126348] = -torch.inf

            x0, ranking_score, token_confidence, raw_entropy, normalized_entropy, uncertainty_token_confidence = _sample_tokens_and_scores(
                score_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                base_alg=attention_discount_base_alg,
                uncertainty_from_filtered_probs=(sampler_variant != "eb_entropy"),
            )

            ranking_score[:, prompt_len + (num_block + 1) * block_length :] = -np.inf
            token_confidence[:, prompt_len + (num_block + 1) * block_length :] = -np.inf
            uncertainty_token_confidence[:, prompt_len + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, ranking_score, -torch.inf)
            selected_confidence = torch.where(mask_index, token_confidence, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            step_details = [None for _ in range(prompt.shape[0])]
            for batch_idx in range(confidence.shape[0]):
                if not bool(active_samples[batch_idx]):
                    continue

                selection_mask_row = torch.zeros_like(mask_index[batch_idx])
                selection_mask_row[block_slice_start:block_slice_end] = active_block_mask[batch_idx]
                current_mask_count = int(selection_mask_row.sum().item())
                if current_mask_count <= 0:
                    continue

                baseline_num_select = (
                    int(num_transfer_tokens[batch_idx, step_idx].item())
                    if num_transfer_tokens is not None
                    else current_mask_count
                )
                if baseline_num_select <= 0 and sampler_variant in {"attention_discounted", "confidence_topk"}:
                    continue

                if sampler_variant == "attention_discounted":
                    details = _select_with_attention_discount(
                        confidence_row=confidence[batch_idx],
                        selected_confidence_row=selected_confidence[batch_idx],
                        masked_row=selection_mask_row,
                        attention_matrix=attention[batch_idx],
                        num_select=baseline_num_select,
                        alpha=attention_discount_alpha,
                        mode=attention_discount_mode,
                        score_rule=attention_discount_score_rule,
                        normalization=attention_discount_normalization,
                        aggregation=attention_discount_aggregation,
                        selected_uncertainty=attention_discount_selected_uncertainty,
                        generation_scope=generation_scope,
                        return_details=True,
                    )
                elif sampler_variant == "confidence_topk":
                    selected = _select_top_confidence(
                        confidence_row=confidence[batch_idx],
                        masked_row=selection_mask_row,
                        num_select=baseline_num_select,
                    )
                    details = _empty_selection_details(selected)
                    details["selected_count"] = int(selected.sum().item())
                elif sampler_variant == "attention_discount_ab_budget_eb_stop":
                    details = _select_with_attention_discount_ab_budget_eb_stop(
                        confidence_row=confidence[batch_idx],
                        selected_confidence_row=selected_confidence[batch_idx],
                        entropy_row=raw_entropy[batch_idx],
                        masked_row=selection_mask_row,
                        attention_matrix=attention[batch_idx],
                        max_select=current_mask_count,
                        gamma=dbes_gamma,
                        alpha=attention_discount_alpha,
                        mode=attention_discount_mode,
                        score_rule=attention_discount_score_rule,
                        normalization=attention_discount_normalization,
                        aggregation=attention_discount_aggregation,
                        selected_uncertainty=attention_discount_selected_uncertainty,
                        generation_scope=generation_scope,
                        return_details=True,
                    )
                elif sampler_variant == "fast_dllm":
                    details = _select_with_fast_dllm(
                        confidence_row=confidence[batch_idx],
                        masked_row=selection_mask_row,
                        gamma=dbes_gamma,
                        return_details=True,
                    )
                elif sampler_variant == "ad_fast_dllm":
                    details = _select_with_attention_discount_fast_dllm(
                        confidence_row=confidence[batch_idx],
                        selected_confidence_row=selected_confidence[batch_idx],
                        masked_row=selection_mask_row,
                        attention_matrix=attention[batch_idx],
                        max_select=current_mask_count,
                        gamma=dbes_gamma,
                        alpha=attention_discount_alpha,
                        mode=attention_discount_mode,
                        score_rule=attention_discount_score_rule,
                        normalization=attention_discount_normalization,
                        aggregation=attention_discount_aggregation,
                        selected_uncertainty=attention_discount_selected_uncertainty,
                        generation_scope=generation_scope,
                        return_details=True,
                    )
                else:
                    details = _select_with_eb_paper_exact(
                        error_proxy_row=-confidence[batch_idx],
                        entropy_row=raw_entropy[batch_idx],
                        masked_row=selection_mask_row,
                        gamma=dbes_gamma,
                        return_details=True,
                    )
                details["selected_count"] = int(details["selected"].sum().item())
                if details["selected_count"] == 0:
                    raise RuntimeError(
                        f"{sampler_variant} selected zero tokens on an active block; increase gamma or use a different configuration."
                    )
                transfer_index[batch_idx] = details["selected"]
                step_details[batch_idx] = details

            x[transfer_index] = x0[transfer_index]
            for sample_idx, details in enumerate(step_details):
                if details is None:
                    continue
                selected_counts_per_sample[sample_idx].append(int(details["selected_count"]))
                selected_risks_per_sample[sample_idx].extend(float(v) for v in details["selected_risks"])
                step_budget_values_per_sample[sample_idx].extend(float(v) for v in details["budget_values"])
                selected_confidences_per_sample[sample_idx].extend(
                    None if v is None else float(v) for v in details["accepted_confidences"]
                )
                selected_uncertainty_weights_per_sample[sample_idx].extend(
                    None if v is None else float(v) for v in details["accepted_uncertainty_weights"]
                )
                selected_raw_dependencies_per_sample[sample_idx].extend(
                    float(v) for v in details["accepted_raw_dependencies"]
                )
                selection_overlap_ratio_per_sample[sample_idx].append(details.get("selection_overlap_ratio"))
                selection_replacement_count_per_sample[sample_idx].append(details.get("selection_replacement_count"))
                selected_base_score_mean_per_sample[sample_idx].append(details.get("selected_base_score_mean"))
                selected_effective_score_mean_per_sample[sample_idx].append(details.get("selected_effective_score_mean"))
                selected_score_shift_mean_per_sample[sample_idx].append(details.get("selected_score_shift_mean"))

    if return_stats:
        stats = {
            "nfe_per_sample": [int(v) for v in nfe_per_sample.tolist()],
            "selected_counts_per_sample": selected_counts_per_sample,
            "selected_risks_per_sample": selected_risks_per_sample,
            "step_budget_values_per_sample": step_budget_values_per_sample,
            "selected_confidences_per_sample": selected_confidences_per_sample,
            "selected_uncertainty_weights_per_sample": selected_uncertainty_weights_per_sample,
            "selected_raw_dependencies_per_sample": selected_raw_dependencies_per_sample,
            "selection_overlap_ratio_per_sample": selection_overlap_ratio_per_sample,
            "selection_replacement_count_per_sample": selection_replacement_count_per_sample,
            "selected_base_score_mean_per_sample": selected_base_score_mean_per_sample,
            "selected_effective_score_mean_per_sample": selected_effective_score_mean_per_sample,
            "selected_score_shift_mean_per_sample": selected_score_shift_mean_per_sample,
        }
        return x, stats

    return x
