import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def bipartite_soft_matching_randframe(metric: torch.Tensor, 
                                      F: int, ratio: float, unm_pre: int, generator: torch.Generator,
                                      target_stride: int = 4, align_batch: bool = False,
                                      merge_mode: str = "replace") -> Tuple[Callable, Callable, dict]:
    """
    Partitions the multi-frame tokens into src and dst and merges ratio of src tokens from src to dst.
    Dst tokens are partitioned by choosing one random frame.

    Args:
        - metric [B, N, C]: metric to use for similarity.
        - F: frame number.
        - ratio: ratio of src tokens to be removed (by merging).
        - unm_pre: number of src tokens not merged at previous ToMe. Pre-sequence: [unm_pre|F_0|F_1|...]
        - generator: random number generator
        - target_stride: stride of target frame.
        - align_batch: whether to align similarity matching maps of samples in the batch. True when using PnP.
        - merge_mode: how to merge tokens. "mean": tokens -> Mean(src_token, dst_token); "replace": tokens -> dst_token.

    Returns:
        Merge and unmerge operation according to the matching result. Return a dict including other values.
    """
    B, N, _ = metric.shape
    tnum = (N - unm_pre) // F

    if ratio <= 0:
        return do_nothing, do_nothing, {"unm_num": tnum}

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        idx_buffer = torch.arange(
            N - unm_pre, device=metric.device, dtype=torch.int64)

        target_stride = min(target_stride, F)
        randf = torch.randint(0, target_stride, torch.Size(
            [1]), generator=generator, device=generator.device)
        dst_select = ((torch.div(idx_buffer, tnum, rounding_mode='floor')) %
                      target_stride == randf).to(torch.bool)

        a_idx = idx_buffer[None, ~dst_select, None] + unm_pre
        b_idx = idx_buffer[None, dst_select, None] + unm_pre

        unm_buffer = torch.arange(unm_pre, device=metric.device, dtype=torch.int64)[
            None, :, None]
        b_idx = torch.cat([b_idx, unm_buffer], dim=1)

        del idx_buffer, unm_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        r = min(a.shape[1], int(a.shape[1] * ratio))


        if align_batch:
            scores = torch.cat([*scores], dim=-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None],
                             dim=-2, index=src_idx) % num_dst # Map index to (0, num_dst - 1)
            
            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode=None) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c),
                                     src, reduce=mode, include_self=True)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, **kwarg) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=s_idx).expand(-1, -1, c), src=src)

        return out


    ret_dict = {"unm_num": unm_idx.shape[1] if unm_idx.shape[1] is not None else 0}
    return merge, unmerge, ret_dict


def bipartite_soft_matching_clip_adaptive(
    metric: torch.Tensor, F: int, ratio: float, unm_pre: int, generator: torch.Generator,
    target_stride: int = 4, align_batch: bool = False,
    merge_mode: str = "weighted", weight_threshold: float = 0.8,
    anchor_frame_idx: int = 0,
    token_mapping: torch.Tensor = None,
    C: int = 768,
    spatial_window: int = 3,  
    temporal_window_radius: int = 1,  
    frame_height: int = 64,  
    frame_width: int = 64,  
) -> Tuple[Callable, Callable, dict]:

    B, N, _ = metric.shape
    if N - unm_pre <= 0 or F <= 0:
        return do_nothing, do_nothing, {"unm_num": 0}

    tnum = frame_height * frame_width 
    if tnum <= 0 or (N - unm_pre) // F != tnum:
        return do_nothing, do_nothing, {"unm_num": tnum}

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    device = metric.device

    with torch.no_grad():
        idx_buffer = torch.arange(N - unm_pre, device=device, dtype=torch.int64).reshape(F, tnum)
        dst_select = torch.zeros(F, tnum, dtype=torch.bool, device=device)
        dst_select[anchor_frame_idx] = True
        dst_select = dst_select.flatten()

        a_idx = idx_buffer[~dst_select].reshape(1, -1, 1) + unm_pre  
        b_idx = idx_buffer[dst_select].reshape(1, -1, 1) + unm_pre    

        unm_buffer = torch.arange(unm_pre, device=device, dtype=torch.int64)[None, :, None]
        b_idx = torch.cat([b_idx, unm_buffer], dim=1)

        del idx_buffer, unm_buffer
        num_dst = b_idx.shape[1]
        num_src = a_idx.shape[1]

        src_frame_ids = (a_idx[0, :, 0] - unm_pre) // tnum
        src_flat_spatial_ids = (a_idx[0, :, 0] - unm_pre) % tnum
        src_y = src_flat_spatial_ids // frame_width
        src_x = src_flat_spatial_ids % frame_width

        dst_frame_ids = (b_idx[0, :num_dst-unm_pre, 0] - unm_pre) // tnum
        dst_flat_spatial_ids = (b_idx[0, :num_dst-unm_pre, 0] - unm_pre) % tnum
        dst_y = dst_flat_spatial_ids // frame_width
        dst_x = dst_flat_spatial_ids % frame_width

        temporal_mask = torch.abs(src_frame_ids.unsqueeze(1) - dst_frame_ids.unsqueeze(0)) <= temporal_window_radius
        spatial_mask = (torch.abs(src_y.unsqueeze(1) - dst_y.unsqueeze(0)) <= spatial_window//2) & \
                       (torch.abs(src_x.unsqueeze(1) - dst_x.unsqueeze(0)) <= spatial_window//2)
        valid_mask = temporal_mask & spatial_mask

        edge_frame_mask = (src_frame_ids == 0) | (src_frame_ids == F-1)
        if edge_frame_mask.any():
            src_frame_ids[src_frame_ids == 0] = 1  
            src_frame_ids[src_frame_ids == F-1] = F-2 

        metric_norm = metric / (metric.norm(dim=-1, keepdim=True) + 1e-8)
        src = gather(metric_norm, dim=1, index=a_idx.expand(B, num_src, C))  # [B, num_src, C]
        dst = gather(metric_norm, dim=1, index=b_idx.expand(B, num_dst, C))  # [B, num_dst, C]


        src_float = src.float()
        dst_float = dst.float()
        l1_dist = torch.cdist(src_float, dst_float, p=1)  # [B, num_src, num_dst]
        l1_dist = l1_dist.to(src.dtype)
        sim = 1.0 - torch.clamp(l1_dist / C, 0.0, 1.0)
        sim = sim * valid_mask.unsqueeze(0)  
        sim = torch.where(sim > weight_threshold, sim, torch.zeros_like(sim))  

        node_max, node_idx = sim.max(dim=-1)
        merge_mask = node_max > 0.0
        unm_mask = ~merge_mask

        edge_idx = torch.arange(num_src, device=device, dtype=torch.int64)[None, :, None].expand(B, -1, -1)
        src_idx = edge_idx[merge_mask.unsqueeze(-1).expand(-1, -1, 1)].view(B, -1, 1)
        unm_idx = edge_idx[unm_mask.unsqueeze(-1).expand(-1, -1, 1)].view(B, -1, 1)
        dst_idx = gather(node_idx.unsqueeze(-1), dim=1, index=src_idx)

        if align_batch and B > 1:
            src_idx = src_idx[:1].expand(B, -1, -1)
            unm_idx = unm_idx[:1].expand(B, -1, -1)
            dst_idx = dst_idx[:1].expand(B, -1, -1)

    def merge(x: torch.Tensor) -> torch.Tensor:
        src_full, dst_full = split(x)
        B_src, N_src, C_src = src_full.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src_full, dim=-2, index=u_idx.expand(B_src, -1, C_src))
        selected_src = gather(src_full, dim=-2, index=s_idx.expand(B_src, -1, C_src))
        selected_sim = gather(node_max[..., None], dim=-2, index=s_idx.expand(B_src, -1, 1))

        for b in range(B_src):
            unique_dst, inverse_indices = torch.unique(d_idx[b], return_inverse=True)
            for i, dst_pos in enumerate(unique_dst):
                mask = inverse_indices == i
                src_tokens = selected_src[b, mask]
                src_sims = selected_sim[b, mask]
                

                sum_sim = src_sims.sum()
                sum_feat = (src_sims.unsqueeze(-1) * src_tokens).sum(dim=0)
                dst_full[b, dst_pos] = (dst_full[b, dst_pos] + sum_feat) / (1 + sum_sim)

        return torch.cat([unm, dst_full], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        B_unm, _, C_unm = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        out = torch.zeros(B_unm, N, C_unm, device=device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B_unm, -1, C_unm), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B_unm, -1, 1), dim=1, index=u_idx).expand(-1, -1, C_unm), src=unm)
        
        if token_mapping is not None:
            for f in range(F):
                frame_mapping = token_mapping[f]  # [H, W, 2]
                anchor_coords = frame_mapping[..., 0].long()
                merge_weights = frame_mapping[..., 1].unsqueeze(-1)
                frame_token_idx = a_idx[0, (a_idx[0,:,0]-unm_pre)//tnum == f, 0]
                if frame_token_idx.numel() == 0:
                    continue
                anchor_token_idx = b_idx[0, anchor_coords.flatten(), 0]
                out[:, frame_token_idx] = out[:, anchor_token_idx] * merge_weights.flatten().unsqueeze(0)
        else:
            src_restored = gather(dst, dim=-2, index=d_idx.expand(B_unm, -1, C_unm))
            out.scatter_(dim=-2, index=gather(a_idx.expand(B_unm, -1, 1), dim=1, index=s_idx).expand(-1, -1, C_unm), src=src_restored)

        return out

    def split(x):
        b, n, c = x.shape
        src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
        dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
        return src, dst

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict



def bipartite_soft_matching_random2d_hier(metric: torch.Tensor, frame_num: int, ratio: float, unm_pre: int, generator: torch.Generator, target_stride: int = 4, adhere_src: bool = False,  merge_mode: str = "replace", scores = None, coord = None, rec_field = 2) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape
    F = frame_num
    nf = (N - unm_pre) // F

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():

        
      
        idx_buffer = torch.arange(N - unm_pre, device=metric.device, dtype=torch.int64)


        max_f = min(target_stride, F)
        randn = torch.randint(0, max_f, torch.Size([1]), generator=generator, device = generator.device)
        dst_select = ((torch.div(idx_buffer, nf, rounding_mode='floor')) % max_f == randn).to(torch.bool)
        a_idx = idx_buffer[None, ~dst_select, None] + unm_pre
        b_idx = idx_buffer[None, dst_select, None] + unm_pre

        unm_buffer = torch.arange(unm_pre, device=metric.device, dtype=torch.int64)[None,:,None]
        b_idx = torch.cat([b_idx, unm_buffer], dim = 1)


        del idx_buffer, unm_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst
        
        def split_coord(coord):
            b, n, c = coord.shape
            src = gather(coord, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(coord, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst


        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        

        if coord is not None:
            src_coord, dst_coord = split_coord(coord)
            mask = torch.norm(src_coord[:,:,None,:] - dst_coord[:,None,:,:], dim=-1) > rec_field
            
        
        scores = a @ b.transpose(-1, -2)

        if coord is not None:
            scores[mask] = 0


        r = int(a.shape[1] * ratio)
        r = min(a.shape[1], r)



        if adhere_src:
            scores = torch.cat([*scores], dim = -1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  
            src_idx = edge_idx[..., :r, :]  
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)


    def merge(x: torch.Tensor, mode=None, b_select = None,  **kwarg) -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        
        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c), src, reduce=mode, include_self=True)


        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, b_select = None, unm_modi = None,  **kwarg) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        if unm_modi is not None:
            if unm_modi == "zero":
                unm = torch.zeros_like(unm)
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=s_idx).expand(-1, -1, c), src=src)

        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict


def bipartite_soft_matching_2s( metric: torch.Tensor, 
                                src_len: int, ratio: float, align_batch: bool,
                                merge_mode: str = "replace", unmerge_chunk: int = 0) -> Tuple[Callable, Callable, dict]:

    B, N, _ = metric.shape

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():

        idx_buffer = torch.arange(N, device=metric.device, dtype=torch.int64)

        a_idx = idx_buffer[None, :src_len, None]
        b_idx = idx_buffer[None, src_len:, None]

        del idx_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        scores = a @ b.transpose(-1, -2)

        r = min(a.shape[1], int(a.shape[1] * ratio))

        if align_batch:
            scores = torch.cat([*scores], dim=-1)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  
            src_idx = edge_idx[..., :r, :]  
            dst_idx = gather(node_idx[..., None],
                             dim=-2, index=src_idx) % num_dst 
            
    
            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode=None) -> torch.Tensor:

        src, dst = split(x)
        n, t1, c = src.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))

            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c),
                                     src, reduce=mode, include_self=True)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, **kwarg) -> torch.Tensor:

        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx

        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=u_idx).expand(-1, -1, c), src=unm)

        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1),
                     dim=1, index=s_idx).expand(-1, -1, c), src=src)
        
        out = out[:, :src_len, :] if unmerge_chunk == 0 else out[:, src_len:, :]
        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict


# Original ToMe
def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:

    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        if no_rand:
            rand_idx = torch.zeros(
                hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(
                sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

    
        idx_buffer_view = torch.zeros(
            hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(
            dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(
            hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)


        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(
                h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        del idx_buffer, idx_buffer_view


        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst


        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        r = min(a.shape[1], r)
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B,
                     a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B,
                     a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching_2f(metric: torch.Tensor, src_len: int, ratio: float, adhere_src: bool, merge_mode: str = "replace", scores = None, coord = None, rec_field = 2, unmerge_chunk = 0) -> Tuple[Callable, Callable]:

    B, N, _ = metric.shape

    if ratio <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():

        
        idx_buffer = torch.arange(N, device=metric.device, dtype=torch.int64)
        a_idx = idx_buffer[None, :src_len, None]
        b_idx = idx_buffer[None, src_len:, None]



        del idx_buffer

        num_dst = b_idx.shape[1]

        def split(x):
            b, n, c = x.shape
            src = gather(x, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(x, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst
        
        def split_coord(coord):
            b, n, c = coord.shape
            src = gather(coord, dim=1, index=a_idx.expand(b, n - num_dst, c))
            dst = gather(coord, dim=1, index=b_idx.expand(b, num_dst, c))
            return src, dst


        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        

        if coord is not None:
            src_coord, dst_coord = split_coord(coord)
            mask = torch.norm(src_coord[:,:,None,:] - dst_coord[:,None,:,:], dim=-1) > rec_field
            
        
        scores = a @ b.transpose(-1, -2)

        if coord is not None:
            scores[mask] = 0

        r = int(a.shape[1] * ratio)
        r = min(a.shape[1], r)



        if adhere_src:
            scores = torch.cat([*scores], dim = -1)
            # scores = torch.sum(scores, dim=0)
            node_max, node_idx = scores.max(dim=-1)

            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  
            src_idx = edge_idx[..., :r, :]  
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx) % num_dst

            unm_idx = unm_idx.expand(B, -1, -1)
            src_idx = src_idx.expand(B, -1, -1)
            dst_idx = dst_idx.expand(B, -1, -1)
        else:

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)



    def merge(x: torch.Tensor, mode=None, b_select = None) -> torch.Tensor:

        src, dst = split(x)
        n, t1, c = src.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        
        unm = gather(src, dim=-2, index=u_idx.expand(-1, -1, c))
        # src = gather(src, dim=-2, index=s_idx.expand(-1, -1, c))
        mode = mode if mode is not None else merge_mode
        if mode != "replace":
            dst = dst.scatter_reduce(-2, d_idx.expand(-1, -1, c), src, reduce=mode, include_self=True)


        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor, b_select = None, unm_modi = None) -> torch.Tensor:



        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        b, _, c = unm.shape
        if b_select is not None:
            if not isinstance(b_select, list):
                b_select = [b_select]
            u_idx, s_idx, d_idx = unm_idx[b_select], src_idx[b_select], dst_idx[b_select]
        else:
            u_idx, s_idx, d_idx = unm_idx, src_idx, dst_idx
        if unm_modi is not None:
            if unm_modi == "zero":
                unm = torch.zeros_like(unm)
        src = gather(dst, dim=-2, index=d_idx.expand(-1, -1, c))

        # Combine back to the original shape
        out = torch.zeros(b, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(b, -1, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=u_idx).expand(-1, -1, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(b, -1, 1), dim=1, index=s_idx).expand(-1, -1, c), src=src)

        
        if unmerge_chunk == 0:
            out = out[:,:src_len,:]
        else:
            out = out[:,src_len:,:]

        return out

    ret_dict = {"unm_num": unm_idx.shape[1]}
    return merge, unmerge, ret_dict