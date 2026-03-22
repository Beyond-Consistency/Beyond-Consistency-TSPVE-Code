import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from transformers import logging
import random

from utils import CONTROLNET_DICT
from utils import load_config, save_config
from utils import get_controlnet_kwargs, get_frame_ids, get_latents_dir, init_model, seed_everything
from utils import prepare_control, load_latent, load_video, prepare_depth, save_video
from utils import register_time, register_attention_control, register_conv_control

from utils.utils import save_video_with_ffmpeg
import TSPVE
from utils.dift_sd import SDFeaturizer
from TSPVE.patch import update_patch  
import matplotlib.pyplot as plt  
from sklearn.manifold import TSNE  
from torch_geometric.utils import to_undirected, add_self_loops, degree  
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
logging.set_verbosity_error()


class Generator(nn.Module):
    def __init__(self, pipe, scheduler, config):
        super().__init__()

        self.device = config.device
        self.seed = config.seed
        self.model_key = config.model_key
        self.config = config
        
        gene_config = config.generation
        float_precision = gene_config.float_precision if "float_precision" in gene_config else config.float_precision
        self.dtype = torch.float16 if float_precision == "fp16" else torch.float32
        print(f"[INFO] float precision {float_precision}. Use {self.dtype}.")

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        
        if config.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ModuleNotFoundError:
                print("[WARNING] xformers not found. Disable xformers attention.")

        self.n_timesteps = gene_config.n_timesteps
        scheduler.set_timesteps(gene_config.n_timesteps, device=self.device)
        self.scheduler = scheduler

        self.batch_size = 2
        self.control = gene_config.control
        self.use_depth = config.sd_version == "depth"
        self.use_controlnet = self.control in CONTROLNET_DICT.keys()
        self.use_pnp = self.control == "pnp"
        
        if self.use_controlnet:
            self.controlnet = pipe.controlnet
            self.controlnet_scale = gene_config.control_scale
        elif self.use_pnp:
            pnp_f_t = int(gene_config.n_timesteps * gene_config.pnp_f_t)
            pnp_attn_t = int(gene_config.n_timesteps * gene_config.pnp_attn_t)
            self.batch_size += 1
            self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)

        self.chunk_size = gene_config.chunk_size
        self.enable_acac = gene_config.get("enable_acac", False)
        self.merge_global = not self.enable_acac 
        self.local_merge_ratio = gene_config.local_merge_ratio
        self.global_merge_ratio = gene_config.global_merge_ratio
        self.global_rand = gene_config.global_rand
        self.align_batch = gene_config.align_batch
        
        self.prompt = gene_config.prompt
        self.negative_prompt = gene_config.negative_prompt
        self.guidance_scale = gene_config.guidance_scale
        self.save_frame = gene_config.save_frame
        self.visualize_pca = gene_config.get("visualize_pca", False)
        
        self.frame_height, self.frame_width = config.height, config.width
        self.work_dir = config.work_dir

        self.dift_model = None
        self.dift_features = None
        
        self.base_merge_ratio = gene_config.get("base_ratio", 0.8)  
        self.temporal_smooth = gene_config.get("temporal_smooth", 0.01)  
        self.min_merge_ratio = gene_config.get("min_ratio", 0.5)    
        self.max_merge_ratio = gene_config.get("max_ratio", 0.9)    
        
        self.prev_merge_ratio = self.base_merge_ratio
        self.keyframe_similarities = []
        self.acac_weight = 1.0  

        self.dift_config = {
            "H": self.frame_height,
            "W": self.frame_width,
            "max_batch_size": gene_config.get("max_batch_size", 8),
            "similarity_mean_threshold": gene_config.get("similarity_mean_threshold", 0.9),
            "similarity_region_threshold": gene_config.get("similarity_region_threshold", 0.9),
            "max_kv_size": gene_config.get("max_kv_size", 8),
            "output_path": self.work_dir,
            "C": 1280  
        }


        self.feature_stride = config.get("feature_stride", 256)  
        assert self.feature_stride >= 8, "feature_stride must be ≥8"

        self.matching = {}
        self.batch_begin_idx = []
        self.dift_size = []
        self.key_matching = {}
        self.all_pivotal_idx = {}
        self.dift_features = []
        self.anchor_frames = []
        self.clip_frame_ranges = []
        self.token_mapping = {}  
        self.preprocessed = False
        

        self.activate_TSPVE()

        if gene_config.use_lora:
            self.pipe.load_lora_weights(**gene_config.lora)

    def activate_TSPVE(self):
        TSPVE.apply_patch(
            self.pipe,
            self.local_merge_ratio,
            self.merge_global,
            self.global_merge_ratio, 
            seed=self.seed,
            batch_size=self.batch_size,
            align_batch=self.use_pnp or self.align_batch,
            global_rand=self.global_rand,
            clip_adaptive=True
        )        

    def load_frames_and_preprocess(self, data_path, frame_ids, model_key):
        self.frames = load_video(
            data_path, self.frame_height, self.frame_width, frame_ids=frame_ids, device=self.device
        ).to(torch.float16).to(self.device)
        
        self.extract_dift_features(model_key)
        self.compute_batch_segments()
        self.all_pivotal_idx = self.generate_pivotal(self.batch_begin_idx)
        self.compute_resolution_matching(model_key)
        self.preprocessed = True

    def extract_dift_features(self, model_key):
        dift = SDFeaturizer(model_key)
        stride_to_index = {32: 0, 16: 1, 8: 2, 4: 3}
        feature_index = stride_to_index.get(self.feature_stride, 2)
        
        H = self.frame_height // self.feature_stride  
        W = self.frame_width // self.feature_stride   
        self.dift_size = [(H, W)]  
        
        ft_list = []  
        for i in tqdm(range(len(self.frames)), desc='Extracting DIFT features'):
            ft = dift.forward(
                self.frames[i], 
                prompt='', 
                t=0, 
                up_ft_index=[feature_index], 
                ensemble_size=8
            )
            ft_list.append(ft[feature_index].detach().to(self.device))  
        self.dift_features = torch.cat(ft_list) if ft_list else torch.tensor([]).to(self.device)
        
        if self.dift_features.numel() > 0:
            self.C = self.dift_features.shape[1]
            gcn1_input_dim = self.C + 2  
        else:
            self.C = 640  
            gcn1_input_dim = self.C + 2

    def compute_token_similarity(self, idx1, idx2):
        f1 = self.dift_features[idx1].view(self.C, -1).T  # [H*W, C]
        f2 = self.dift_features[idx2].view(self.C, -1).T
        f1 = F.normalize(f1, dim=-1)
        f2 = F.normalize(f2, dim=-1)
        sim = torch.matmul(f1, f2.T)  # [N, N]
        sim_max = sim.max(dim=1).values
        return sim_max.mean().item(), sim_max.reshape(self.frame_height // self.feature_stride, -1).mean().item()

    def compute_batch_segments(self):
        h_m = self.dift_config.get("segment_threshold", 0.6)
        max_kv = self.dift_config.get("max_batch_size", 8)
        self.batch_begin_idx = [0]

        if self.config.get("batch_size", "auto") == "auto":
            n = len(self.frames)
            cur_s = 0
            while cur_s < n:
                next_s = cur_s + 1
                while next_s < n:
                    mean_sim, _ = self.compute_token_similarity(cur_s, next_s)
                    if mean_sim < h_m or next_s - cur_s >= max_kv:
                        break
                    next_s += 1

                if next_s < n:
                    self.batch_begin_idx.append(next_s)
                cur_s = next_s

            if self.batch_begin_idx[-1] != n:
                self.batch_begin_idx.append(n)
        else:
            batch_size = int(self.config.get("batch_size", 1))
            self.batch_begin_idx = list(range(0, len(self.frames), batch_size))
            if self.batch_begin_idx[-1] != len(self.frames):
                self.batch_begin_idx.append(len(self.frames))

        self.batch_begin_idx = sorted(set(self.batch_begin_idx))
        if self.batch_begin_idx[-1] != len(self.frames):
            self.batch_begin_idx.append(len(self.frames))

        self.clip_frame_ranges = []
        for i in range(len(self.batch_begin_idx) - 1):
            self.clip_frame_ranges.append((self.batch_begin_idx[i], self.batch_begin_idx[i + 1]))

        print(f'Batch begin index (preprocessed): {self.batch_begin_idx}')
        print(f'Clip frame ranges: {self.clip_frame_ranges}')

    def compute_resolution_matching(self, model_key):
        device = self.device
        for (h, w) in tqdm(self.dift_size, desc='Matching DIFT features', total=len(self.dift_size)):
            frames_ft = self.dift_features
            frames_ft = nn.functional.interpolate(frames_ft, size=(h, w), mode='bilinear', align_corners=False)
            frames_ft = frames_ft.permute(0, 2, 3, 1)  # [n_frames, h, w, c]

            for i in range(len(self.batch_begin_idx) - 1):
                frame_idx = self.batch_begin_idx[i]
                batch_size = self.batch_begin_idx[i+1] - frame_idx
                if batch_size <= 0:
                    continue
                self.matching[f'{h}_{w}_{i}'] = self.get_matching(
                    frames_ft, frame_idx, batch_size, device=device
                ).detach().cpu()

            torch.cuda.empty_cache()

            if len(self.batch_begin_idx) > self.config["max_kv_size"]:
                for t in self.scheduler.timesteps:
                    t = int(t)
                    key_frames_ft = frames_ft[self.all_pivotal_idx[t]]
                    key_match = self.get_matching(
                        key_frames_ft, 0, len(key_frames_ft), is_key=True, device=device
                    )
                    k = int(key_match.shape[1] // (key_match.shape[0] / self.config["max_kv_size"]))
                    self.key_matching[f'{h}_{w}_{t}'] = torch.topk(
                        key_match, k, dim=1, largest=True
                    ).indices.detach().cpu()

    def get_matching(self, frames_ft, begin_idx, batch_size, is_key=False, device='cuda'):
        dim = frames_ft.shape[-1]
        if batch_size <= self.config["max_kv_size"]:
            batch_ft = frames_ft[begin_idx:begin_idx+batch_size].reshape(-1, dim).to(device)
            batch_ft = batch_ft / batch_ft.norm(dim=-1, keepdim=True)
            similarity = batch_ft @ batch_ft.T
            sim_list = similarity.chunk(batch_size, dim=1)
            idx = []
            for sim in sim_list:
                idx.append(sim.argmax(dim=1).unsqueeze(0) if not is_key else sim.max(dim=1).values.unsqueeze(0))
        else:
            small_batch_size = self.config["max_batch_size"] // batch_size
            idx = []
            for j in range(0, batch_size, small_batch_size):
                batch_ft_j = frames_ft[begin_idx+j:begin_idx+min(j+small_batch_size, batch_size)]
                batch_ft_j = batch_ft_j.reshape(-1, dim).to(device)
                batch_ft_j = batch_ft_j / batch_ft_j.norm(dim=-1, keepdim=True)
                similarity = batch_ft_j @ frames_ft[begin_idx:begin_idx+batch_size].reshape(-1, dim).T.to(device)
                sim_list = similarity.chunk(batch_size, dim=1)
                
                if len(idx) == 0:
                    for sim in sim_list:
                        idx.append(sim.argmax(dim=1).unsqueeze(0) if not is_key else sim.max(dim=1).values.unsqueeze(0))
                else:
                    for k, sim in enumerate(sim_list):
                        idx[k] = torch.cat([idx[k], sim.argmax(dim=1).unsqueeze(0)], dim=1) if not is_key else \
                                 torch.cat([idx[k], sim.max(dim=1).values.unsqueeze(0)], dim=1)
        return torch.cat(idx)

    def check_similarity(self, sim_tensor, region_size, stride):
        if torch.mean(sim_tensor) < self.config["similarity_mean_threshold"]:
            print(f"Mean similarity is too low: {torch.mean(sim_tensor)}")
            return False
        
        for i in range(0, sim_tensor.shape[0] - region_size + 1, stride):
            for j in range(0, sim_tensor.shape[1] - region_size + 1, stride):
                region_mean = torch.mean(sim_tensor[i:i+region_size, j:j+region_size])
                if region_mean < self.config["similarity_region_threshold"]:
                    print(f"Region mean similarity is too low: {region_mean}")
                    return False
        return True

    def generate_pivotal(self, batch_begin_idx):
        anchor_indices = []
        for i in range(len(batch_begin_idx) - 1):
            start = batch_begin_idx[i]
            end = batch_begin_idx[i + 1]
            segment_dift = self.dift_features[start:end]  # [K, C, H, W]
            frame_features = segment_dift.mean(dim=(2, 3))  # [K, C]
            frame_features = F.normalize(frame_features, dim=1)

            similarity_matrix = torch.mm(frame_features, frame_features.T)  # [K, K]
            relevance_scores = similarity_matrix.sum(dim=1)
            anchor_offset = relevance_scores.argmax().item()
            anchor_indices.append(start + anchor_offset)

        self.anchor_frames = anchor_indices


        all_pivotal_idx = {}
        for t in self.scheduler.timesteps:
            t_int = int(t)
            all_pivotal_idx[t_int] = torch.tensor(self.anchor_frames, device=self.device)
        return all_pivotal_idx

    @torch.no_grad()
    def get_text_embeds_input(self, prompt, negative_prompt):
        text_embeds = self.get_text_embeds(prompt, negative_prompt, self.device)
        if self.use_pnp:
            pnp_guidance_embeds = self.get_text_embeds("", device=self.device)
            text_embeds = torch.cat([pnp_guidance_embeds, text_embeds], dim=0)
        return text_embeds

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt=None, device="cuda"):
        text_input = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors='pt'
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        
        if negative_prompt is not None:
            uncond_input = self.tokenizer(
                negative_prompt, 
                padding='max_length', 
                max_length=self.tokenizer.model_max_length,
                truncation=True, 
                return_tensors='pt'
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def prepare_data(self, data_path, latent_path, frame_ids):
        self.frames = load_video(
            data_path, self.frame_height, self.frame_width, frame_ids=frame_ids, device=self.device
        ).to(self.dtype)
        self.init_noise = load_latent(
            latent_path, t=self.scheduler.timesteps[0], frame_ids=frame_ids
        ).to(self.dtype).to(self.device)
        
        if self.use_depth:
            self.depths = prepare_depth(
                self.pipe, self.frames, frame_ids, self.work_dir
            ).to(self.dtype).to(self.device)
        
        if self.use_controlnet:
            self.controlnet_images = prepare_control(
                self.control, self.frames, frame_ids, self.work_dir
            ).to(self.dtype).to(self.device)
        
        assert self.batch_begin_idx, "Preprocessed batch_begin_idx is empty!"
        assert self.all_pivotal_idx, "Preprocessed all_pivotal_idx is empty!"

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def decode_latents_batch(self, latents):
        imgs = []
        for latent in latents.split(self.batch_size, dim=0):
            imgs.append(self.decode_latents(latent))
        return torch.cat(imgs)

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def encode_imgs_batch(self, imgs):
        latents = []
        for img in imgs.split(self.batch_size, dim=0):
            latents.append(self.encode_imgs(img))
        return torch.cat(latents)
    
    def get_chunks(self, flen, time_step_index):
        chunks = []
        for i in range(len(self.clip_frame_ranges)):
            clip_start, clip_end = self.clip_frame_ranges[i]
            for j in range(0, clip_end - clip_start, self.chunk_size):
                chunk_start = clip_start + j
                chunk_end = min(chunk_start + self.chunk_size, clip_end)
                if chunk_start < chunk_end:
                    chunks.append(torch.arange(chunk_start, chunk_end, device=self.device))
        return chunks
        

    @torch.no_grad()
    def ddim_sample(self, x, conds):
        print("[INFO] denoising frames...")
        timesteps = self.scheduler.timesteps
        noises = torch.zeros_like(x)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            self.pre_iter(x, t)
            chunks = self.get_chunks(len(x), time_step_index=i)
            
            if self.merge_global and i == 0:
                self.segment_p_raws = []
                self.processed_segments = set()
            
            current_timestep_int = int(t.item())
            self.pipe.acac_step = i
            self.pipe.acac_total_steps = len(timesteps)
            for chunk in chunks:
                self.update_merge_ratio(frame_indices=chunk, current_timestep=current_timestep_int, step=i)
                noises[chunk] = self.pred_noise(x[chunk], conds, t, batch_idx=chunk)
            
            x = self.pred_next_x(x, noises, t, i, inversion=False)
            self.post_iter(x, t)
        
        return x

    def pre_iter(self, x, t):
        if self.use_pnp:
            register_time(self, t.item())
            self.cur_latents = load_latent(self.latent_path, t=t, frame_ids=self.frame_ids)
        
        if self.dift_features.size(2) > 0 and self.dift_features.size(3) > 0:
            self.current_resolution = (self.dift_features.size(2), self.dift_features.size(3))
        else:
            self.current_resolution = (self.frame_height // 8, self.frame_width // 8)

    def post_iter(self, x, t):
        if self.merge_global:
            TSPVE.update_patch(self.pipe, global_tokens=None)

    @torch.no_grad()
    def pred_noise(self, x, cond, t, batch_idx=None):
        flen = len(x)
        text_embed_input = cond.repeat_interleave(flen, dim=0)
        latent_model_input = torch.cat([x, x])
        batch_size = 2

        if self.use_pnp:
            source_latents = self.cur_latents[batch_idx] if batch_idx is not None else self.cur_latents
            latent_model_input = torch.cat([source_latents.to(x), latent_model_input])
            batch_size += 1

        if self.use_depth:
            depth = self.depths[batch_idx] if batch_idx is not None else self.depths
            depth = depth.repeat(batch_size, 1, 1, 1)
            latent_model_input = torch.cat([latent_model_input, depth.to(x)], dim=1)
        
        if self.use_controlnet:
            controlnet_cond = self.controlnet_images[batch_idx] if batch_idx is not None else self.controlnet_images
            controlnet_cond = controlnet_cond.repeat(batch_size, 1, 1, 1)
            kwargs = get_controlnet_kwargs(
                self.controlnet, latent_model_input, text_embed_input, t, controlnet_cond, self.controlnet_scale
            )
        else:
            kwargs = {}
        
        eps = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, **kwargs).sample
        noise_pred_uncond, noise_pred_cond = eps.chunk(batch_size)[-2:]
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    @torch.no_grad()
    def pred_next_x(self, x, eps, t, i, inversion=False):
        timesteps = reversed(self.scheduler.timesteps) if inversion else self.scheduler.timesteps
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        
        if inversion:
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[timesteps[i-1]] if i > 0 else self.scheduler.final_alpha_cumprod
        else:
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else self.scheduler.final_alpha_cumprod
        
        mu = alpha_prod_t ** 0.5
        sigma = (1 - alpha_prod_t) ** 0.5
        mu_prev = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
        
        pred_x0 = (x - sigma * eps) / mu if not inversion else (x - sigma_prev * eps) / mu_prev
        x = mu_prev * pred_x0 + sigma_prev * eps if not inversion else mu * pred_x0 + sigma * eps
        return x

    def init_pnp(self, conv_injection_t, qk_injection_t):
        qk_steps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        conv_steps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control(self, qk_steps, num_inputs=self.batch_size)
        register_conv_control(self, conv_steps, num_inputs=self.batch_size)

    def check_latent_exists(self, latent_path):
        timesteps = self.scheduler.timesteps if self.use_pnp else [self.scheduler.timesteps[0]]
        for ts in timesteps:
            if not os.path.exists(os.path.join(latent_path, f'noisy_latents_{ts}.pt')):
                return False
        return True

    def get_current_key_features(self, t):
        h, w = self.current_resolution
        h = min([64, 32, 16, 8], key=lambda x: abs(x - h))
        w = min([64, 32, 16, 8], key=lambda x: abs(x - w))
        key = f"{h}_{w}_{int(t)}"
        return self.dift_features[self.all_pivotal_idx[int(t)]] if key not in self.key_matching else \
               self.dift_features[self.key_matching[key]]

    def calculate_pairwise_similarity(self, features):
        if features is None:
            return torch.tensor([0.5], device=self.device)
        
        if features.ndim == 4:
            features = features.view(features.size(0), -1, features.size(1))
            features = features.mean(dim=1)
        elif features.ndim == 3:
            features = features.mean(dim=1)
        
        features = F.normalize(features, dim=-1)
        return torch.mm(features, features.T)[torch.triu_indices(features.size(0), features.size(1), offset=1)]

    def get_segment_frames(self, segment_id):
        if segment_id >= len(self.batch_begin_idx):
            raise IndexError(f"Segment ID {segment_id} out of range (max {len(self.batch_begin_idx)-1})")
        start = self.batch_begin_idx[segment_id]
        end = self.batch_begin_idx[segment_id + 1] if segment_id + 1 < len(self.batch_begin_idx) else len(self.frames)
        return torch.arange(start, end, device=self.device)

    def find_segment_id(self, frame_indices):
        for i in range(len(self.batch_begin_idx) - 1):
            if (frame_indices >= self.batch_begin_idx[i]).all() and (frame_indices < self.batch_begin_idx[i+1]).all():
                return i

        if len(self.batch_begin_idx) > 1:
            return len(self.batch_begin_idx) - 2
        return 0

    def update_merge_ratio(self, frame_indices, current_timestep, step):
        local_ratio = self.compute_adaptive_ratio(frame_indices, current_timestep)

        total_steps = self.n_timesteps
        acac_info = {}
        acac_enabled = False
        if self.enable_acac:
            if step >= total_steps - 10:
                acac_info = self.acac_cross_clip_attention(step) or {}
                acac_enabled = bool(acac_info.get("clip_pairs"))
        segment_id = self.find_segment_id(frame_indices)
        current_anchor_idx = None
        current_token_mapping = None
        clip_frame_indices = None
        if segment_id is not None and len(self.anchor_frames) > segment_id:
            clip_start = self.batch_begin_idx[segment_id]
            clip_end = self.batch_begin_idx[segment_id + 1]
            current_anchor_idx = self.anchor_frames[segment_id] - clip_start
            clip_frame_indices = list(range(clip_start, clip_end))
            mapping_key = tuple(frame_indices.tolist())
            current_token_mapping = self.token_mapping.get(mapping_key, None)

        TSPVE.update_patch(
            self.pipe,
            local_merge_ratio=local_ratio,
            global_merge_ratio=self.global_merge_ratio if not self.enable_acac else 0.0,
            merge_global=not self.enable_acac,
            current_anchor_idx=current_anchor_idx,
            current_token_mapping=current_token_mapping,
            feature_dim=self.C,
            clip_frame_indices=clip_frame_indices,
            acac_enabled=acac_enabled,
            acac_clip_pairs=acac_info.get("clip_pairs", []),
            acac_clip_frame_ranges=acac_info.get("clip_frame_ranges", self.clip_frame_ranges), 
            acac_weight=acac_info.get("weight", 0.0),
            acac_step=step,
            acac_total_steps=total_steps,
            acac_frame_indices=frame_indices.tolist()
        )


    def get_time_embedding(self, frame_indices, current_timestep):
        T = len(self.frames)
        emb_dim = 8
        frame_indices = frame_indices.to(self.device).float()
        emb = torch.zeros((len(frame_indices), emb_dim), device=self.device)
        
        for i in range(emb_dim // 2):
            emb[:, 2*i] = torch.sin(frame_indices * (10000 ** (2*i/emb_dim)))
            emb[:, 2*i+1] = torch.cos(frame_indices * (10000 ** (2*i/emb_dim)))
        return emb

    def acac_cross_clip_attention(self, current_step):
        if not hasattr(self, 'anchor_frames') or len(self.anchor_frames) < 2:
            return None

        clip_pairs = []
        n_real_clips = len(self.clip_frame_ranges)  
        if current_step % 2 == 0:
            for i in range(0, n_real_clips - 1, 2):
                clip_pairs.append((i, i + 1))
        else:
            for i in range(1, n_real_clips - 1, 2):
                clip_pairs.append((i, i + 1))

        if not clip_pairs:
            return None

        frame_tokens = self.dift_features.mean(dim=(2, 3))
        pair_similarities = []
        valid_clip_pairs = []
        for (clip_a_id, clip_b_id) in clip_pairs:
            if clip_a_id >= len(self.anchor_frames) or clip_b_id >= len(self.anchor_frames):
                continue
            anchor_a = self.anchor_frames[clip_a_id]
            anchor_b = self.anchor_frames[clip_b_id]
            feat_a = F.normalize(frame_tokens[anchor_a].unsqueeze(0), dim=-1)
            feat_b = F.normalize(frame_tokens[anchor_b].unsqueeze(0), dim=-1)
            sim = torch.matmul(feat_a, feat_b.T).item()
            pair_similarities.append(sim)
            valid_clip_pairs.append((clip_a_id, clip_b_id))

        if not valid_clip_pairs:
            return None

        mean_sim = float(sum(pair_similarities) / len(pair_similarities))
        acac_weight = 0.2 + (mean_sim * 0.6)
        acac_weight = torch.clamp(torch.tensor(acac_weight), 0.0, 1.0).item()

    
        
        return {
            "clip_pairs": valid_clip_pairs,
            "clip_frame_ranges": self.clip_frame_ranges, 
            "weight": acac_weight,
            "mean_similarity": mean_sim
        }

    def build_clip_adaptive_mapping(self, frame_indices):
        if len(frame_indices) == 0:
            return None

        base_h = self.frame_height // self.feature_stride
        base_w = self.frame_width // self.feature_stride
        N = base_h * base_w

        anchor_idx = self.anchor_frames[self.find_segment_id(frame_indices)]
        anchor_ft = self.dift_features[anchor_idx] # [C,h,w]

        mapping = torch.zeros((len(frame_indices), base_h, base_w, 2), device=self.device)

        for fi_idx, f in enumerate(frame_indices):
            frame_ft = self.dift_features[f]
            for y in range(base_h):
                for x in range(base_w):
                    neigh = []
                    for dt in [-1, 0, 1]:
                        f_int = int(f.item()) if torch.is_tensor(f) else int(f)
                        t_idx = min(max(f_int + dt, 0), len(self.frames) - 1)
                        ft_t = self.dift_features[t_idx]
                        for dy in [-1,0,1]:
                            for dx in [-1,0,1]:
                                yy = min(max(y+dy, 0), base_h-1)
                                xx = min(max(x+dx, 0), base_w-1)
                                neigh.append(ft_t[:,yy,xx])

                    token = frame_ft[:,y,x]
                    token_repeat = token.unsqueeze(0).expand(len(neigh), -1)
                    neigh_stack = torch.stack(neigh, dim=0)

                    l1 = torch.abs(token_repeat - neigh_stack).sum(dim=-1)
                    sim = 1 - (l1 / token.shape[0])
                    mask = (sim > 0.8).float()
                    if mask.sum() > 0:
                        weight = (sim * mask).sum().item() / mask.sum().item()
                    else:
                        weight = 0.0
                    mapping[fi_idx, y, x, 0] = anchor_idx
                    mapping[fi_idx, y, x, 1] = weight

        self.token_mapping[tuple(frame_indices.tolist())] = mapping
        return mapping

    def compute_adaptive_ratio(self, frame_indices, current_timestep):
        if current_timestep not in self.all_pivotal_idx or len(frame_indices)==0:
            return self.base_merge_ratio

        mapping = self.build_clip_adaptive_mapping(frame_indices)
        if mapping is None:
            return self.base_merge_ratio

        mean_weight = mapping[:,:,:,1].mean().item()
        ratio = 1.0 - mean_weight
        local_ratio = 0.5 + (mean_weight * 0.4)
        local_ratio = max(min(local_ratio, self.max_merge_ratio), self.min_merge_ratio)
        return local_ratio

    def compute_pca_model(self, features):


        features = features.detach().cpu()
        num_frames, C, H, W = features.shape
        feat_2d = features.permute(0, 2, 3, 1).reshape(-1, C).numpy()
        pca = PCA(n_components=3)
        pca.fit(feat_2d)
        return pca

    # def features_to_pca_rgb(self, features, pca):

    #     features = features.detach().cpu()
    #     num_frames, C, H, W = features.shape
    #     feat_2d = features.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    #     feat_pca = pca.transform(feat_2d)
    #     feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
    #     feat_pca_img = feat_pca.reshape(num_frames, H, W, 3)
    #     return feat_pca_img

    # def show_pca_frames(self, pca_imgs, save_path=None, title="PCA Feature Maps"):
    #     """
    #     pca_imgs: [num_frames, H, W, 3]
    #     """
    #     num_frames = pca_imgs.shape[0]
    #     fig, axes = plt.subplots(1, min(4, num_frames), figsize=(16, 4))
    #     for i in range(min(4, num_frames)):
    #         axes[i].imshow(pca_imgs[i])
    #         axes[i].set_title(f"Frame {i}")
    #         axes[i].axis('off')
    #     plt.suptitle(title)
    #     if save_path:
    #         plt.savefig(save_path)
    #     plt.show()

    def show_xt_slice(self, pca_imgs, save_path=None, title="x-t slice"):
        """
        pca_imgs: [num_frames, H, W, 3]
        """
        y = pca_imgs.shape[1] // 2  
        xt_slice = pca_imgs[:, y, :, :]  # [num_frames, W, 3]
        xt_slice_img = np.transpose(xt_slice, (1, 0, 2))  # [W, num_frames, 3]
        plt.figure(figsize=(12, 3))
        plt.imshow(xt_slice_img)
        plt.title(title)
        plt.xlabel("t (frame)")
        plt.ylabel("x")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @torch.no_grad()
    @torch.no_grad()
    def __call__(self, data_path, latent_path, output_path, frame_ids):
        self.scheduler.set_timesteps(self.n_timesteps)
        latent_path = get_latents_dir(latent_path, self.model_key)
        assert self.check_latent_exists(latent_path), "Required latent not found."
        
        self.data_path = data_path
        self.latent_path = latent_path
        self.frame_ids = frame_ids

        if not self.preprocessed:
            self.load_frames_and_preprocess(data_path, frame_ids, self.model_key)

        self.prepare_data(data_path, latent_path, frame_ids)
        self.current_resolution = (self.frame_height // 8, self.frame_width // 8)
        
        print(f"[INFO] initial noise latent shape: {self.init_noise.shape}")
        
        src_features = self.dift_features.detach().clone()
        pca = self.compute_pca_model(src_features)
        if self.visualize_pca:
            self.show_pca_frames(
                src_pca_imgs, 
                save_path=os.path.join(output_path, "src_pca_rgb.png"), 
                title="Source PCA Features"
            )
            self.show_xt_slice(
                src_pca_imgs, 
                save_path=os.path.join(output_path, "src_xt_slice.png"), 
                title="Source x-t slice"
            )

        for edit_name, edit_prompt in self.prompt.items():
            print(f"[INFO] current prompt: {edit_prompt}")
            conds = self.get_text_embeds_input(edit_prompt, self.negative_prompt)
            clean_latent = self.ddim_sample(self.init_noise, conds)
            torch.cuda.empty_cache()
            
            clean_frames = self.decode_latents_batch(clean_latent)
            cur_output_path = os.path.join(output_path, edit_name)
            save_config(self.config, cur_output_path, gene=True)
            #save_video(clean_frames, cur_output_path, save_frame=self.save_frame)
            save_video_with_ffmpeg(clean_frames, cur_output_path)

    def extract_dift_features_for_imgs(self, imgs):

        dift = SDFeaturizer(self.model_key)
        stride_to_index = {32: 0, 16: 1, 8: 2, 4: 3}
        feature_index = stride_to_index.get(self.feature_stride, 2)
        ft_list = []
        for i in range(imgs.shape[0]):
            ft = dift.forward(
                imgs[i].to(self.device), 
                prompt='', 
                t=0, 
                up_ft_index=[feature_index], 
                ensemble_size=8
            )
            ft_list.append(ft[feature_index].detach().to(self.device))
        return torch.cat(ft_list) if ft_list else torch.tensor([]).to(self.device)


if __name__ == "__main__":
    config = load_config()
    pipe, scheduler, model_key = init_model(
        config.device, 
        config.sd_version, 
        config.model_key, 
        config.generation.control, 
        config.float_precision
    )
    config.model_key = model_key
    seed_everything(config.seed)
    
    generator = Generator(pipe, scheduler, config)
    frame_ids = get_frame_ids(
        config.generation.frame_range, 
        config.generation.frame_ids
    )
    generator(
        config.input_path, 
        config.generation.latents_path,
        config.generation.output_path, 
        frame_ids=frame_ids
    )