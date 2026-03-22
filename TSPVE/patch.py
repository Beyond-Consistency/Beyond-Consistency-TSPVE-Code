import math
import time
from typing import Type, Dict, Any, Tuple, Callable

import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F

from . import merge
from .utils import isinstance_str, init_generator, join_frame, split_frame, func_warper, join_warper, split_warper


def compute_merge(module: torch.nn.Module, x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    generator = module.generator

    fsize = x.shape[0] // args["batch_size"]
    tsize = x.shape[1]

    if downsample <= args["max_downsample"]:
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        local_merge_ratio = getattr(module, "local_merge_ratio", args["local_merge_ratio"])
        anchor_frame_idx = getattr(module, "current_anchor_idx", None)
        token_mapping = getattr(module, "current_token_mapping", None)
        C = getattr(module, "feature_dim", 768)
        clip_frame_indices = getattr(module, "clip_frame_indices", None)
        
        if args.get("clip_adaptive", False) and anchor_frame_idx is not None and clip_frame_indices is not None:
            return _compute_iterative_merging(
                x, fsize, tsize, args, generator,
                anchor_frame_idx, token_mapping, C, clip_frame_indices
            )
        
        local_tokens = join_frame(x, fsize)
        m_ls = [join_warper(fsize)]
        u_ls = [split_warper(fsize)]
        unm = 0
        curF = fsize

        while curF > 1:
            if args.get("clip_adaptive", False):
                m, u, ret_dict = merge.bipartite_soft_matching_clip_adaptive(
                    local_tokens, curF, local_merge_ratio, unm, generator,
                    args["target_stride"], args["align_batch"],
                    anchor_frame_idx=anchor_frame_idx,
                    token_mapping=token_mapping,
                    C=C
                )
            else:
                m, u, ret_dict = merge.bipartite_soft_matching_randframe(
                    local_tokens, curF, local_merge_ratio, unm, generator,
                    args["target_stride"], args["align_batch"])
            unm += ret_dict["unm_num"]
            m_ls.append(m)
            u_ls.append(u)
            local_tokens = m(local_tokens)
            curF = (local_tokens.shape[1] - unm) // tsize

        merged_tokens = local_tokens
        if args["merge_global"]:
            pass 

        m = func_warper(m_ls)
        u = func_warper(u_ls[::-1])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)
        merged_tokens = x

    return m, u, merged_tokens


def _compute_iterative_merging(
    x: torch.Tensor, fsize: int, tsize: int, args: Dict, generator: torch.Generator,
    anchor_frame_idx: int, token_mapping: torch.Tensor, C: int, clip_frame_indices: list,
    frame_height: int = 64, frame_width: int = 64
):

    device = x.device
    B, N, C_feat = x.shape
    clip_len = len(clip_frame_indices)
    if clip_len <= 1:
        return merge.do_nothing, merge.do_nothing, x

    anchor_rel_idx = anchor_frame_idx
    merge_state = {
        "merged_frames": [anchor_rel_idx],
        "left_ptr": anchor_rel_idx - 1,
        "right_ptr": anchor_rel_idx + 1
    }
    merge_history = []
    unmerge_history = []
    current_tokens = x.clone()
    total_unm = 0

    while merge_state["left_ptr"] >= 0 or merge_state["right_ptr"] < clip_len:
        frames_to_merge = []
        if merge_state["left_ptr"] >= 0:
            frames_to_merge.append(merge_state["left_ptr"])
            merge_state["left_ptr"] -= 1
        if merge_state["right_ptr"] < clip_len:
            frames_to_merge.append(merge_state["right_ptr"])
            merge_state["right_ptr"] += 1
        if not frames_to_merge:
            break

        current_merge_frames = merge_state["merged_frames"] + frames_to_merge
        current_F = len(current_merge_frames)
        new_anchor_idx = 0

        m_step, u_step, ret_dict = merge.bipartite_soft_matching_clip_adaptive(
            current_tokens, current_F, 0.0, total_unm, generator,
            args["target_stride"], args["align_batch"],
            anchor_frame_idx=new_anchor_idx,
            token_mapping=token_mapping,
            C=C,
            frame_height=frame_height,
            frame_width=frame_width
        )

        merge_history.append(m_step)
        unmerge_history.append(u_step)
        total_unm += ret_dict["unm_num"]
        merge_state["merged_frames"] = current_merge_frames
        current_tokens = m_step(current_tokens)

    def final_merge(x_in):
        res = x_in
        for m in merge_history:
            res = m(res)
        return res

    def final_unmerge(x_in):
        res = x_in
        for u in reversed(unmerge_history):
            res = u(res)
        return res

    return final_merge, final_unmerge, current_tokens


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(
                self, x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)),
                    context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x

    return ToMeBlock


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            m_a, u_a, merged_tokens = compute_merge(
                self, norm_hidden_states, self._tome_info)
            acac_input = merged_tokens  

            acac_enabled = getattr(self, "acac_enabled", False)
            acac_clip_pairs = getattr(self, "acac_clip_pairs", [])
            acac_clip_frame_ranges = getattr(self, "acac_clip_frame_ranges", [])
            acac_weight = getattr(self, "acac_weight", 0.0)
            acac_frame_indices = getattr(self, "acac_frame_indices", None)

            if acac_enabled and acac_weight > 0.0 and len(acac_clip_pairs) > 0 and len(acac_clip_frame_ranges) > 0 and acac_frame_indices is not None:
                try:
                    device = acac_input.device
                    dtype = acac_input.dtype
                    B_total, N_tokens, C_feat = acac_input.shape
                    
                    batch_size_base = getattr(self._tome_info["args"], "batch_size", 2)
                    F_actual = len(acac_frame_indices)
                    if B_total >= F_actual:
                        actual_merged = acac_input[-F_actual:]
                    else:
                        actual_merged = acac_input

                    frame_to_token_idx = {f: i for i, f in enumerate(acac_frame_indices)}
                    updated_merged = actual_merged.clone()

                    relevant_clip_pairs = []
                    for (clip_a_id, clip_b_id) in acac_clip_pairs:
                        if clip_a_id >= len(acac_clip_frame_ranges) or clip_b_id >= len(acac_clip_frame_ranges):
                            continue
                        
                        a_start, a_end = acac_clip_frame_ranges[clip_a_id]
                        b_start, b_end = acac_clip_frame_ranges[clip_b_id]
                        
                        a_has_frames = any(f in frame_to_token_idx for f in range(a_start, a_end))
                        b_has_frames = any(f in frame_to_token_idx for f in range(b_start, b_end))
                        
                        if a_has_frames and b_has_frames: 
                            relevant_clip_pairs.append((clip_a_id, clip_b_id))

                    for (clip_a_id, clip_b_id) in relevant_clip_pairs:
                        a_start, a_end = acac_clip_frame_ranges[clip_a_id]
                        b_start, b_end = acac_clip_frame_ranges[clip_b_id]
                        
                        clip_a_frames_in_chunk = [f for f in acac_frame_indices if a_start <= f < a_end]
                        clip_b_frames_in_chunk = [f for f in acac_frame_indices if b_start <= f < b_end]
                        
                        if not clip_a_frames_in_chunk or not clip_b_frames_in_chunk:
                            continue

                        clip_a_token_idx = [frame_to_token_idx[f] for f in clip_a_frames_in_chunk]
                        clip_b_token_idx = [frame_to_token_idx[f] for f in clip_b_frames_in_chunk]
                        
                        tokens_a = actual_merged[clip_a_token_idx]
                        tokens_b = actual_merged[clip_b_token_idx]

                        tokens_a_flat = tokens_a.reshape(-1, C_feat)
                        tokens_b_flat = tokens_b.reshape(-1, C_feat)
                        joint_tokens = torch.cat([tokens_a_flat, tokens_b_flat], dim=0)

                        q = k = v = F.normalize(joint_tokens, dim=-1)
                        d_k = C_feat
                        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
                        attn_weights = torch.softmax(attn_scores, dim=-1)
                        attn_output = torch.matmul(attn_weights, v)

                        len_a = tokens_a_flat.shape[0]
                        attn_a = attn_output[:len_a].reshape(tokens_a.shape)
                        attn_b = attn_output[len_a:].reshape(tokens_b.shape)
                        
                        updated_merged[clip_a_token_idx] = updated_merged[clip_a_token_idx] * (1 - acac_weight) + attn_a * acac_weight
                        updated_merged[clip_b_token_idx] = updated_merged[clip_b_token_idx] * (1 - acac_weight) + attn_b * acac_weight

                    if B_total >= F_actual:
                        acac_input[-F_actual:] = updated_merged
                    else:
                        acac_input = updated_merged
                    
                    merged_tokens = acac_input

                except Exception as e:
                    pass

            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                merged_tokens, 
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            attn_output = u_a(attn_output)
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            norm_hidden_states = self.norm3(hidden_states)
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            
            hidden_states = ff_output + hidden_states
            
            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_tome_module(module: torch.nn.Module):

    def hook(module, args):
        if not hasattr(module, "generator"):
            module.generator = init_generator(args[0].device)
        elif module.generator.device != args[0].device:
            module.generator = init_generator(
                args[0].device, fallback=module.generator)
        else:
            return None

        # module.generator = module.generator.manual_seed(module._tome_info["args"]["seed"])
        return None

    module._tome_info["hooks"].append(module.register_forward_pre_hook(hook))


def apply_patch(
        model: torch.nn.Module,
        local_merge_ratio: float = 0.9,
        merge_global: bool = False,
        global_merge_ratio=0.8,
        max_downsample: int = 2,
        seed: int = 123,
        batch_size: int = 2,
        include_control: bool = False,
        align_batch: bool = False,
        target_stride: int = 4,
        global_rand=0.5,
        clip_adaptive: bool = False):
    

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(
        model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError(
                "Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    if isinstance_str(model, "StableDiffusionControlNetPipeline") and include_control:
        diffusion_models = [diffusion_model, model.controlnet]
    else:
        diffusion_models = [diffusion_model]

    for diffusion_model in diffusion_models:
        use_global = merge_global and (not clip_adaptive)
        diffusion_model._tome_info = {
            "size": None,
            "hooks": [],
            "args": {
                "max_downsample": max_downsample,
                "generator": None,
                "seed": seed,
                "batch_size": batch_size,
                "align_batch": align_batch,
                "merge_global": use_global,
                "global_merge_ratio": global_merge_ratio,
                "local_merge_ratio": local_merge_ratio,
                "global_rand": global_rand,
                "target_stride": target_stride,
                "clip_adaptive": clip_adaptive,
                "acac_enabled": False,
                "acac_pairs": [],
                "acac_weight": 0.0,
                "acac_step": 0,
                "acac_total_steps": 0
            }
        }
        hook_tome_model(diffusion_model)

        for name, module in diffusion_model.named_modules():
            # If for some reason this has a different name, create an issue and I'll fix it
            # if isinstance_str(module, "BasicTransformerBlock") and "down_blocks" not in name:
            if isinstance_str(module, "BasicTransformerBlock"):
                make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
                module.__class__ = make_tome_block_fn(module.__class__)
                module._tome_info = diffusion_model._tome_info
                hook_tome_module(module)

                # Something introduced in SD 2.0 (LDM only)
                if not hasattr(module, "disable_self_attn") and not is_diffusers:
                    module.disable_self_attn = False

                # Something needed for older versions of diffusers
                if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers

    model = model.unet if hasattr(model, "unet") else model
    model_ls = [model]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for hook in module._tome_info["hooks"]:
                    hook.remove()
                module._tome_info["hooks"].clear()

            if module.__class__.__name__ == "ToMeBlock":
                module.__class__ = module._parent

    return model


def update_patch(model: torch.nn.Module, **kwargs):
    """ Update arguments in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    for model in model_ls:
        for _, module in model.named_modules():
            if hasattr(module, "_tome_info"):
                for k, v in kwargs.items():
                    setattr(module, k, v)
    return model


def collect_from_patch(model: torch.nn.Module, attr="tome"):
    """ Collect attributes in patched modules """
    # For diffusers
    model0 = model.unet if hasattr(model, "unet") else model
    model_ls = [model0]
    if hasattr(model, "controlnet"):
        model_ls.append(model.controlnet)
    ret_dict = dict()
    for model in model_ls:
        for name, module in model.named_modules():
            if hasattr(module, attr):
                res = getattr(module, attr)
                ret_dict[name] = res

    return ret_dict
