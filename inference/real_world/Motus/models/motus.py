# Motus - Modular Architecture
# Three-modal UniDiffuser: Video Model (WAN) + Action Expert + Understanding Expert
# Implements MoT (Mixture of Tokens) architecture with unified attention

import sys
import json
import torch
import logging
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

BAK_ROOT = str((Path(__file__).parent.parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from utils.common import get_t_distribution
from wan.modules.model import sinusoidal_embedding_1d
from transformers import Qwen3VLForConditionalGeneration, AutoConfig
from .wan_model import WanVideoModel
from .action_expert import ActionExpert, ActionExpertConfig
from .und_expert import UndExpert, UndExpertConfig
# Add Flow-Matching schedulers
from wan.utils.fm import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)

@dataclass 
class MotusConfig:
    """Configuration for Motus."""
    # Video model settings
    wan_checkpoint_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    vae_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    wan_config_path: str = "/share/home/bhz/pretrained_models/Wan2.2-TI2V-5B"
    video_precision: str = "bfloat16"

    # VLM settings
    vlm_checkpoint_path: str = "/share/home/bhz/pretrained_models/Qwen3-VL-2B-Instruct"
    
    # Understanding Expert settings - configurable from yaml
    und_expert_hidden_size: int = 512        # Understanding expert hidden dimension
    und_expert_ffn_dim_multiplier: int = 4   # Understanding expert FFN dimension multiplier
    und_expert_norm_eps: float = 1e-5        # Understanding expert layer norm epsilon
    und_layers_to_extract: List[int] = None  # Which VLM layers to extract from
    
    # VLM adapter settings for understanding expert
    vlm_adapter_input_dim: int = 2048        # VLM feature dimension (input)
    vlm_adapter_projector_type: str = "mlp3x_silu"  # VLM adapter type

    # Action expert settings  
    num_layers: int = 30 
    action_state_dim: int = 14
    action_dim: int = 14
    action_expert_dim: int = 1024           # Configurable hidden dimension
    action_expert_ffn_dim_multiplier: int = 4  # FFN dimension multiplier
    action_expert_norm_eps: float = 1e-6    # Layer norm epsilon for Action Expert

    # Sampling settings
    global_downsample_rate: int = 3     # Global downsampling rate
    video_action_freq_ratio: int = 4    # Video:Action frequency ratio
    num_video_frames: int = 4           # Number of video frames to predict
    
    # Video dimensions
    video_height: int = 512             # Input video height
    video_width: int = 512              # Input video width
    
    # Training settings
    batch_size: int = 8

    # Training mode
    training_mode: str = 'finetune'  # 'pretrain' or 'finetune'

    # Loss weights
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    # Control whether to load pretrained WAN/VLM backbones.
    # None = default behavior (load), False = skip loading (init from config only)
    load_pretrained_backbones: Optional[bool] = None

    def __post_init__(self):
        """Calculate derived parameters."""
        # Action chunk size is determined by global downsample rate and frequency ratio
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        
        # Default understanding layers to extract from (if not specified)
        if self.und_layers_to_extract is None:
            # Extract from all layers for comprehensive understanding
            self.und_layers_to_extract = list(range(self.num_layers))


class VideoModule(nn.Module):
    """视频分支模块，负责 WAN 视频 token 处理与 T5 条件注入。

    约定的核心张量维度:
        - 视频 latent: `[B, C_latent, T_latent, H_latent, W_latent]`
        - 视频 token: `[B, L_v, D_wan]`
        - T5 条件: `[B, L_t5, D_t5]` -> 经过 `text_embedding` 后变成 `[B, L_t5, D_wan]`
    """

    def __init__(self, video_model, dtype, device, grid_sizes):
        super().__init__()
        self.video_model = video_model
        self.dtype = dtype
        self.device = device
        self.grid_sizes = grid_sizes

    def prepare_input(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把 noisy video latent 切 patch 后展平成 WAN token。

        参数:
            noisy_video_latent: `[B, C_latent, T_latent, H_latent, W_latent]`

        返回:
            video_features: `[B, L_v, D_wan]`
                - `L_v = T_patch * H_patch * W_patch`
                - `D_wan` 通常为 3072
        """
        # Through patch_embedding: 48 -> 3072 channels
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)  # [B, D_wan, T_patch, H_patch, W_patch]

        # Flatten and convert to tokens
        video_features = video_patched.flatten(2).transpose(1, 2)  # [B, L_v, D_wan]

        # Calculate sequence length and padding
        # seq_lens = torch.tensor([u.size(1) for u in video_tokens_list], dtype=torch.long, device=self.device)
        # seq_len = seq_lens.max().item()

        # Concatenate with padding
        # video_tokens = torch.cat([
        #     torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) 
        #     for u in video_tokens_list
        # ])

        # return video_tokens

        return video_features

    def preprocess_t5_embeddings(self, language_embeddings) -> torch.Tensor:
        """把 T5 文本特征预处理到 WAN 可直接 cross-attn 的空间。

        输入支持两种格式:
            1. `List[Tensor]`，每个元素 `[L_t5_i, D_t5]`
            2. `Tensor[B, L_t5, D_t5]`

        返回:
            t5_context: `[B, L_t5_fixed, D_wan]`
        """
        # Handle both old format (List[torch.Tensor]) and new format (torch.Tensor)
        if isinstance(language_embeddings, list):
            # Old format: List[torch.Tensor] - do padding
            text_len = self.video_model.wan_model.text_len  # 512
            padded_embeddings = []

            for emb in language_embeddings:
                # emb: [L_t5_i, D_t5]
                if emb.shape[0] <= text_len:
                    padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
                else:
                    padded = emb[:text_len]
                padded_embeddings.append(padded)

            t5_context_raw = torch.stack(padded_embeddings, dim=0)  # [B, 512, D_t5]
        else:
            # New format: torch.Tensor [B, seq_len, dim] - already padded by collate_fn
            t5_context_raw = language_embeddings  # [B, L_t5, D_t5]

        # Convert via text_embedding layer (4096 -> 3072)
        t5_context = self.video_model.wan_model.text_embedding(t5_context_raw)  # [B, L_t5_fixed, D_wan]

        return t5_context

    def get_time_embedding(self, t_video: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 WAN 视频分支的时间嵌入与 AdaLN 参数。

        参数:
            t_video: `[B]` 或 `[B, L_v]`
            seq_len: `L_v`

        返回:
            t_emb: `[B, L_v, D_wan]`
            t_emb_proj: `[B, L_v, 6, D_wan]`
                6 份参数分别供 self-attn / FFN 的 AdaLN 调制使用。
        """
        if t_video.dim() == 1:
            t_video = t_video.unsqueeze(1).expand(t_video.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t_video.size(0)
            t_flat = t_video.flatten()  # [B * L_v]
            
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float()
            )  # [B, L_v, D_wan]
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, 3072))  # [B, L_v, 6, D_wan]
            assert t_emb.dtype == torch.float32 and t_emb_proj.dtype == torch.float32
            
        return t_emb, t_emb_proj

    def process_cross_attention(self, video_tokens: torch.Tensor, video_adaln_params: torch.Tensor, 
                               layer_idx: int, processed_t5_context: torch.Tensor) -> torch.Tensor:
        """执行 WAN 与 T5 文本条件之间的 cross-attention。

        参数维度:
            - `video_tokens`: `[B, L_v, D_wan]`
            - `processed_t5_context`: `[B, L_t5, D_wan]`

        返回:
            - 更新后的 `video_tokens`: `[B, L_v, D_wan]`
        """
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        context_lens = None  # WAN uses None for fixed-length context
        cross_out = wan_layer.cross_attn(wan_layer.norm3(video_tokens), processed_t5_context, context_lens)
        return video_tokens + cross_out
    
    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """把预投影后的时间嵌入拆成 6 组 AdaLN 调制参数。

        参数:
            video_adaln_params: `[B, L_v, 6, D_wan]`

        返回:
            长度为 6 的 tuple，每个元素形状都是 `[B, L_v, 1, D_wan]`
        """
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

    def process_ffn(self, video_tokens: torch.Tensor, video_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """执行 WAN 分支本层 FFN。

        参数维度:
            - `video_tokens`: `[B, L_v, D_wan]`
            - `video_adaln_modulation[i]`: `[B, L_v, 1, D_wan]`
        """
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        
        # AdaLN params
        v_mod = video_adaln_modulation

        # WAN FFN with AdaLN (params 3,4,5 for FFN: shift, scale, gate)
        ffn_input = wan_layer.norm2(video_tokens).float() * (1 + v_mod[4].squeeze(2)) + v_mod[3].squeeze(2)  # [B, L_v, D_wan]
        ffn_out = wan_layer.ffn(ffn_input)  # [B, L_v, D_wan]

        with torch.amp.autocast('cuda', dtype=torch.float32):
            return video_tokens + ffn_out * v_mod[5].squeeze(2)

    def apply_output_head(self, video_tokens: torch.Tensor, video_time_emb: torch.Tensor) -> torch.Tensor:
        """把 WAN token 还原成视频 latent 速度场。

        参数:
            - `video_tokens`: `[B, L_v, D_wan]`
            - `video_time_emb`: `[B, L_v, D_wan]`

        返回:
            - `video_velocity`: `[B, C_latent, T_latent, H_latent, W_latent]`
        """
        x = self.video_model.wan_model.head(video_tokens, video_time_emb)  # [B, L_v, patch_dim]
        x = self.video_model.wan_model.unpatchify(x, self.grid_sizes)  # List[[C_latent, T_latent, H_latent, W_latent]]
        return torch.stack([u.float() for u in x], dim=0)  # [B, C_latent, T_latent, H_latent, W_latent]

    def process_joint_attention(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_adaln_modulation: tuple,
        action_adaln_modulation: tuple,
        layer_idx: int,
        action_block: nn.Module,
        und_tokens: torch.Tensor,
        und_block: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """三模态联合 self-attention：视频 token + 动作 token + 理解 token。

        输入维度:
            - `video_tokens`: `[B, L_v, D_wan]`
            - `action_tokens`: `[B, L_a, D_action]`
            - `und_tokens`: `[B, L_u, D_und]`

        输出维度:
            - `video_tokens`: `[B, L_v, D_wan]`
            - `action_tokens`: `[B, L_a, D_action]`
            - `und_tokens`: `[B, L_u, D_und]`
        """
        wan_layer = self.video_model.wan_model.blocks[layer_idx]

        # AdaLN params (already computed)
        v_mod = video_adaln_modulation
        a_mod = action_adaln_modulation

        # Pre-attn normalization with AdaLN
        norm_video = wan_layer.norm1(video_tokens).float() * (1 + v_mod[1].squeeze(2)) + v_mod[0].squeeze(2)  # [B, L_v, D_wan]
        norm_action = action_block.norm1(action_tokens) * (1 + a_mod[1].squeeze(2)) + a_mod[0].squeeze(2)  # [B, L_a, D_action]

        # Get dimensions
        B, L_v, C = norm_video.shape  # C=D_wan
        L_a = norm_action.shape[1]    # 动作 token 数
        n = self.video_model.wan_model.num_heads
        d = C // n                    # 每个 head 的维度

        # Action heads for WAN space (1024 -> 24*128)
        # WAN的动作头权重是固定的，直接用 action_block 的线性层权重来投影动作 token 到 WAN 的 QKV 空间
        a_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_action, action_block.wan_action_qkv)  # [3, B, L_a, n, d]
        a_q_h, a_k_h, a_v_h = a_qkv[0], a_qkv[1], a_qkv[2]
        a_q = action_block.wan_action_norm_q(a_q_h.flatten(-2)).view(B, L_a, n, d)
        a_k = action_block.wan_action_norm_k(a_k_h.flatten(-2)).view(B, L_a, n, d)
        a_v = a_v_h.view(B, L_a, n, d)

        # Understanding Expert processing
        norm_und = und_block.norm1(und_tokens)  # [B, L_u, D_und]
        L_u = norm_und.shape[1]
        
        # Understanding Expert heads for WAN space (2048 -> 24*128)
        u_qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, und_block.wan_und_qkv)  # [3, B, L_u, n, d]
        u_q_h, u_k_h, u_v_h = u_qkv[0], u_qkv[1], u_qkv[2]
        u_q = und_block.wan_und_norm_q(u_q_h.flatten(-2)).view(B, L_u, n, d)
        u_k = und_block.wan_und_norm_k(u_k_h.flatten(-2)).view(B, L_u, n, d)
        u_v = u_v_h.view(B, L_u, n, d)

        # Meta info for WAN attention
        seq_lens = torch.full((B,), L_v + L_a + L_u, dtype=torch.long, device=self.device)  # [B]
        freqs = self.video_model.wan_model.freqs
        if freqs.device != self.device:
            freqs = freqs.to(self.device)

        # Call WAN self-attn with trimodal MoT
        y, action_out_h, und_out_h = wan_layer.self_attn(
            norm_video, seq_lens, self.grid_sizes, freqs,
            action_q=a_q, action_k=a_k, action_v=a_v,
            und_q=u_q, und_k=u_k, und_v=u_v
        )
        
        # Project Understanding Expert output
        und_out = und_block.wan_und_o(und_out_h.flatten(2))  # [B, L_u, D_und]

        # Project back and residual connections
        action_out = action_block.wan_action_o(action_out_h.flatten(2))  # [B, L_a, D_action]
        video_tokens = video_tokens + y * v_mod[2].squeeze(2)
        action_tokens = action_tokens + action_out * a_mod[2].squeeze(2)
        und_tokens = und_tokens + und_out  # Regular residual connection

        return video_tokens, action_tokens, und_tokens


class UndModule(nn.Module):
    """理解分支模块，负责把 Qwen3-VL 的输出转成 UndExpert 可消费的 token。

    核心维度约定:
        - VLM 最后一层隐藏状态: `[B, L_text+L_img, D_vlm]`
        - 适配后理解 token: `[B, L_u, D_und]`
    """

    def __init__(self, vlm_model, und_expert, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        
        # VLM model reference
        self.vlm_model = vlm_model
        
        # Understanding Expert reference
        self.und_expert = und_expert
        
    def extract_und_features(
        self,
        vlm_inputs
    ) -> torch.Tensor:
        """从 VLM 最后一层提取理解特征。

        参数:
            vlm_inputs:
                - 旧格式: `List[Dict]`
                - 新格式: `Dict[str, Tensor]`

        返回:
            adapted_features: `[B, L_u, D_und]`
                其中 `L_u` 实际上就是 VLM 的总序列长度。
        """
        if isinstance(vlm_inputs, list):
            B = len(vlm_inputs)
        else:
            B = vlm_inputs['input_ids'].shape[0]

        # Returns: inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids
        inputs_embeds, attention_mask, visual_pos_masks, deepstack_image_embeds, position_ids = self._process_vlm_inputs_to_tokens(vlm_inputs, B)
        # inputs_embeds: [B, L_u, D_vlm]
        # attention_mask: [B, L_u]
        # visual_pos_masks: [B, L_u]
        # position_ids: [3, B, L_u]

        # Forward through VLM with proper attention_mask and DeepStack features
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # Add DeepStack parameters for Qwen3-VL
        if visual_pos_masks is not None:
            vlm_kwargs['visual_pos_masks'] = visual_pos_masks
        if deepstack_image_embeds is not None:
            vlm_kwargs['deepstack_visual_embeds'] = deepstack_image_embeds

        with torch.no_grad():
            vlm_output = self.vlm_model.model.language_model(**vlm_kwargs)

        # Extract last layer features directly
        last_layer_features = vlm_output.hidden_states[-1]  # [B, L_u, D_vlm]

        # [B, seq_len, vlm_dim] -> [B, seq_len, und_dim]
        # VLM适配器投影到理解专家维度，供后续理解专家处理
        adapted_features = self.und_expert.vlm_adapter(last_layer_features)  # [B, L_u, D_und]

        return adapted_features
        
    def _process_vlm_inputs_to_tokens(self, vlm_inputs, B: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list], torch.Tensor]:
        """Convert VLM inputs to tokens.

        Returns:
            Tuple of:
                - `inputs_embeds`: `[B, L_u, D_vlm]`
                - `attention_mask`: `[B, L_u]`
                - `visual_pos_masks`: `[B, L_u]`
                - `deepstack_image_embeds`: DeepStack 视觉特征列表
                - `position_ids`: `[3, B, L_u]`
        """
        # Handle both old format (List[Dict]) and new format (Dict[str, Tensor])
        if isinstance(vlm_inputs, list):
            # Old format: List[Dict] - do padding and batching
            input_ids_list = [vlm_input['input_ids'] for vlm_input in vlm_inputs]
            attention_mask_list = [vlm_input.get('attention_mask') for vlm_input in vlm_inputs]
            pixel_values_list = [vlm_input.get('pixel_values') for vlm_input in vlm_inputs]
            image_grid_thw_list = [vlm_input.get('image_grid_thw') for vlm_input in vlm_inputs]

            # Pad input_ids and attention_mask to same length
            max_seq_len = max(ids.shape[1] for ids in input_ids_list)
            padded_input_ids = []
            padded_attention_masks = []
            
            for ids, mask in zip(input_ids_list, attention_mask_list):
                if ids.shape[1] < max_seq_len:
                    padding_size = max_seq_len - ids.shape[1]
                    # Pad input_ids with zeros
                    id_padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
                    padded_ids = torch.cat([ids, id_padding], dim=1)
                    # Pad attention_mask with zeros (padding tokens should be ignored)
                    mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                    padded_mask = torch.cat([mask, mask_padding], dim=1)
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)

            # Batch process
            input_ids_batch = torch.cat(padded_input_ids, dim=0).to(self.device)  # [B, L_u]
            attention_mask_batch = torch.cat(padded_attention_masks, dim=0).to(self.device)  # [B, L_u]
            pixel_values_batch = torch.cat([pv.to(self.device) for pv in pixel_values_list], dim=0)  # batch 后的视觉输入
            image_grid_thw_batch = torch.cat([igt.to(self.device) for igt in image_grid_thw_list], dim=0)  # [B, 3]
        else:
            # New format: Dict[str, Tensor] - already batched and padded by collate_fn
            input_ids_batch = vlm_inputs['input_ids'].to(self.device)  # [B, L_u]
            attention_mask_batch = vlm_inputs['attention_mask'].to(self.device)  # [B, L_u]
            pixel_values_batch = vlm_inputs['pixel_values'].to(self.device)
            image_grid_thw_batch = vlm_inputs['image_grid_thw'].to(self.device)  # [B, 3]

        # Get input embeddings
        inputs_embeds = self.vlm_model.get_input_embeddings()(input_ids_batch)  # [B, L_u, D_vlm]

        # Process images - handle different return formats between Qwen2.5-VL and Qwen3-VL
        image_embeds, deepstack_image_embeds = self.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)

        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)  # [N_img_tokens_total, D_vlm]

        # Insert image embeddings
        image_mask, _ = self.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        visual_pos_masks = image_mask[..., 0]  # [B, L_u]，仅图像 token 位置为 True

        # Compute position_ids (position_ids remains as original: [3, B, seq_len])
        # Qwen3-VL get_rope_index has different signature: (input_ids, image_grid_thw, video_grid_thw, attention_mask)
        position_ids, _rope_deltas = self.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None,  # No video in current implementation
            attention_mask=attention_mask_batch
        )

        return inputs_embeds, attention_mask_batch, visual_pos_masks, deepstack_image_embeds, position_ids
    
    def process_ffn(self, und_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """执行理解分支当前层 FFN。

        参数:
            und_tokens: `[B, L_u, D_und]`
        返回:
            `[B, L_u, D_und]`
        """
        block = self.und_expert.blocks[layer_idx]
        
        # Pre-norm for FFN (regular LayerNorm)
        ffn_input = block.norm2(und_tokens)   # [B, L_u, D_und]
        ffn_output = block.ffn(ffn_input)     # [B, L_u, D_und]
        
        # FFN residual connection
        und_tokens = und_tokens + ffn_output
        
        return und_tokens


class ActionModule(nn.Module):
    """动作分支模块，负责 ActionExpert 的时间嵌入、AdaLN 和 FFN。

    约定:
        - 动作 latent / 目标动作: `[B, L_a, D_action_raw]`
        - 动作 token: `[B, L_a(+state/+registers), D_action_hidden]`
    """
    
    def __init__(self, action_expert: ActionExpert, config, video_model, vlm_model, dtype, device):
        super().__init__()
        self.action_expert = action_expert
        self.config = config
        self.video_model = video_model  # For accessing WAN weights
        self.vlm_model = vlm_model      # For accessing VLM weights
        self.dtype = dtype
        self.device = device
    
    def get_time_embedding(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """生成动作分支时间嵌入。

        参数:
            t: `[B]` 或 `[B, L_a]`
            seq_len: 动作 token 长度 `L_a`

        返回:
            - `a_e`: `[B, L_a, D_action_hidden]`
            - `a_e0`: `[B, L_a, 6, D_action_hidden]`
        """
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()  # [B * L_a]
            
            # Create sinusoidal embedding (same pattern as VideoModule)
            #     self.time_embedding = nn.Sequential(
            #     nn.Linear(self.freq_dim, config.dim),
            #     nn.SiLU(),
            #     nn.Linear(config.dim, config.dim)
            # )
            a_e = self.action_expert.time_embedding(
                sinusoidal_embedding_1d(self.action_expert.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float() # 把第0维unflatten成两个维度相乘，shape[0] = bt*seq_len
            )  # [B, L_a, freq_dim] -> [B, L_a, D_action_hidden], config.action_expert_dim通常为1024,freq_dim通常为256
            
            # Project to AdaLN parameters (6 params: 3 for WAN-Action joint attn + 3 for FFN)
            # self.time_projection = nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(config.dim, config.dim * 6)  # 6 parameters: 3 for WAN-Action joint attn + 3 for FFN
            # )
            a_e0 = self.action_expert.time_projection(a_e).unflatten(2, (6, self.config.action_expert_dim))  # [B, L_a, D_action_hidden] =proj=> [B, L_a, 6*D_action_hidden] =unflatten=> [B, L_a, 6, D_action_hidden]
            
            assert a_e.dtype == torch.float32 and a_e0.dtype == torch.float32

        return a_e, a_e0  # (basic_emb, adaln_params)

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """把动作分支的时间投影拆成 6 组 AdaLN 参数。

        参数:
            action_adaln_params: `[B, L_a, 6, D_action_hidden]`
        """
        action_layer = self.action_expert.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0) # [1, 6, 1024] -> [1, 1, 6, 1024],以适配[B, L_a, 6, D_action_hidden]
                + action_adaln_params # [B, L_a, 6, D_action_hidden]
            ).chunk(6, dim=2) # 切成6份，每份对应一个AdaLN参数，得到长度为6的tuple，每个元素形状都是[B, L_a, 1, D_action_hidden]
        return modulation

    def process_ffn(self, action_tokens: torch.Tensor, action_adaln_modulation: tuple, layer_idx: int) -> torch.Tensor:
        """执行动作分支当前层 FFN。

        参数:
            - `action_tokens`: `[B, L_a, D_action_hidden]`
            - `action_adaln_modulation[i]`: `[B, L_a, 1, D_action_hidden]`
        """
        action_block = self.action_expert.blocks[layer_idx]

        # AdaLN params
        a_mod = action_adaln_modulation

        # Apply FFN with AdaLN modulation (params 3,4,5 for FFN: shift, scale, gate)
        ffn_input = action_block.norm2(action_tokens).float() * (1 + a_mod[4].squeeze(2)) + a_mod[3].squeeze(2)  # [B, L_a, D_action_hidden]
        ffn_out = action_block.ffn(ffn_input)  # [B, L_a, D_action_hidden]
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_tokens = action_tokens + ffn_out * a_mod[5].squeeze(2)
        return action_tokens


class Motus(nn.Module):
    """
    Modular Three-modal UniDiffuser with VGM, VLM, and Action modules.
    """

    def __init__(self, config: MotusConfig):
        super().__init__()
        self.config = config

        # Set unified data type for the model
        self.dtype = torch.bfloat16

        # Decide whether to load pretrained backbones
        load_backbones = True if config.load_pretrained_backbones is None else bool(config.load_pretrained_backbones)

        # Initialize video model (WAN)
        logger.info("Initializing WAN video model...")
        if load_backbones:
            self.video_model = WanVideoModel.from_pretrained(
                checkpoint_path=config.wan_checkpoint_path,
                vae_path=config.vae_path,
                config_path=config.wan_config_path,
                precision=config.video_precision
            )
        else:
            self.video_model = WanVideoModel.from_config(
                config_path=config.wan_config_path,
                vae_path=config.vae_path,
                device="cuda",
                precision=config.video_precision
            )

        # Initialize VLM (frozen)
        logger.info("Initializing VLM (frozen)...")
        if load_backbones:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.vlm_checkpoint_path,
                dtype=self.dtype,
                device_map="cuda",
                trust_remote_code=True
            )
        else:
            vlm_cfg = AutoConfig.from_pretrained(config.vlm_checkpoint_path, trust_remote_code=True)
            self.vlm_model = Qwen3VLForConditionalGeneration._from_config(vlm_cfg, torch_dtype=self.dtype)
            self.vlm_model.to(device="cuda", dtype=self.dtype)

        # Freeze VLM parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        logger.info("VLM parameters frozen")

        # Keep VLM complete (do not truncate)
        logger.info(f"VLM kept complete with {len(self.vlm_model.model.language_model.layers)} layers")

        # Get WAN and VLM configurations directly
        wan_dim = getattr(self.video_model.wan_model.config, 'dim', 3072)
        wan_num_heads = getattr(self.video_model.wan_model.config, 'num_heads', 24)
        wan_head_dim = wan_dim // wan_num_heads

        vlm_dim = self.vlm_model.config.text_config.hidden_size
        vlm_num_heads = self.vlm_model.config.text_config.num_attention_heads
        vlm_num_kv_heads = getattr(self.vlm_model.config.text_config if hasattr(self.vlm_model.config, 'text_config') else self.vlm_model.config, 'num_key_value_heads', vlm_num_heads)
        vlm_num_hidden_layers  = self.vlm_model.config.text_config.num_hidden_layers
        vlm_head_dim = vlm_dim // vlm_num_heads

        logger.info(f"Model configurations:")
        logger.info(f"  WAN: {wan_num_heads} heads × {wan_head_dim} head_dim = {wan_dim}D")
        logger.info(f"  VLM: {vlm_num_heads} Q heads, {vlm_num_kv_heads} KV heads × {vlm_head_dim} head_dim = {vlm_dim}D")

        # Create config dictionaries for ActionExpert
        wan_config = {
            'dim': wan_dim,
            'num_heads': wan_num_heads, 
            'head_dim': wan_head_dim
        }
        vlm_config = {
            'hidden_size': vlm_dim,
            'num_attention_heads': vlm_num_heads,
            'num_key_value_heads': vlm_num_kv_heads,
            'head_dim': vlm_head_dim,
            'num_hidden_layers': vlm_num_hidden_layers,
        }

        # Initialize action expert with unified configs
        logger.info("Initializing Action Expert...")

        # Determine chunk_size based on training mode
        if config.training_mode == 'pretrain':
            action_chunk_size_for_expert = config.action_chunk_size
        else:
            action_chunk_size_for_expert = config.action_chunk_size + 1  # include state token

        # Configure registers by mode: no registers in pretrain, keep default (e.g., 4) in finetune
        num_registers = 0 if config.training_mode == 'pretrain' else 4

        action_config = ActionExpertConfig(
            dim=config.action_expert_dim,
            ffn_dim=config.action_expert_dim * config.action_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            state_dim=config.action_state_dim,
            action_dim=config.action_dim,
            chunk_size=action_chunk_size_for_expert,
            num_registers=num_registers,
            video_feature_dim=wan_dim,
            causal=False,
            eps=config.action_expert_norm_eps,
            training_mode=config.training_mode,
        )

        self.action_expert = ActionExpert(action_config, wan_config)

        # Initialize Understanding Expert
        logger.info("Initializing Understanding Expert...")
        und_config = UndExpertConfig(
            dim=config.und_expert_hidden_size,
            ffn_dim=config.und_expert_hidden_size * config.und_expert_ffn_dim_multiplier,
            num_layers=config.num_layers,
            vlm_input_dim=config.vlm_adapter_input_dim,
            vlm_projector_type=config.vlm_adapter_projector_type,
            eps=config.und_expert_norm_eps,
        )
        
        self.und_expert = UndExpert(und_config, wan_config, vlm_config)

        # Move models to device
        self.device = next(self.video_model.parameters()).device
        self.action_expert.to(device=self.device, dtype=self.dtype)
        self.und_expert.to(device=self.device, dtype=self.dtype)
        
        # Set time embedding layers to float32 for numerical stability
        self.action_expert.time_embedding.to(dtype=torch.float32)
        self.action_expert.time_projection.to(dtype=torch.float32)

        # Pre-compute grid_sizes for training batch size
        # VAE的压缩率是(4,16,16)，WAN的patch_size是(1,2,2)，所以总的压缩率是(4,32,32)，latent的T,H,W分别是视频的T,H,W除以压缩率
        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width // 32
        batch_size = config.batch_size
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W], 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3] - pre-expanded
        
        logger.info(f"Pre-computed grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        # Initialize modular components
        self.video_module = VideoModule(self.video_model, self.dtype, self.device, self.grid_sizes)
        self.und_module = UndModule(self.vlm_model, self.und_expert, self.config, self.dtype, self.device)
        self.action_module = ActionModule(self.action_expert, self.config, self.video_model, self.vlm_model, self.dtype, self.device)

        # Initialize t distributions from config
        time_dist_config = getattr(config, 'time_distribution', {})
        model_config = {
            'timestep_sample_method': time_dist_config.get('timestep_sample_method', 'logit_normal'),
            'sigmoid_scale': time_dist_config.get('sigmoid_scale', 1.0),
            'min_t': time_dist_config.get('min_t', 0.0),
            'max_t': time_dist_config.get('max_t', 1.0)
        }

        # Flow-Matching scheduler for training (video branch only)
        try:
            self.fm_train_scheduler = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode to build per-timestep weights (if used)
            self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (video)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler: {e}")

        # Flow-Matching scheduler for training (action branch)
        try:
            self.fm_train_scheduler_action = FlowMatchScheduler(
                shift=5.0,
                sigma_min=0.0,
                extra_one_step=True,
                num_train_timesteps=1000
            )
            # Enable training mode for action as well
            self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)
            logger.info("Initialized FlowMatchScheduler for training (action)")
        except Exception as e:
            logger.warning(f"Failed to init FlowMatchScheduler for action: {e}")

        # Log parameter counts
        self.log_parameter_counts()

    def log_parameter_counts(self):
        """Log detailed parameter counts for each component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        video_params = sum(p.numel() for p in self.video_model.parameters())
        action_params = sum(p.numel() for p in self.action_expert.parameters())
        vlm_params = sum(p.numel() for p in self.vlm_model.parameters())
        und_params = sum(p.numel() for p in self.und_expert.parameters())

        logger.info(f"Motus parameter breakdown:")
        logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Trainable parameters: {trainable_params / 1e9:.2f}B")
        logger.info(f"  Video Model (WAN): {video_params / 1e9:.2f}B")
        logger.info(f"  Action Expert: {action_params / 1e6:.1f}M")
        logger.info(f"  VLM (frozen): {vlm_params / 1e9:.2f}B")
        logger.info(f"  Und Expert: {und_params / 1e6:.1f}M")

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict:
        """Load model checkpoint."""
        # Handle directory path
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)
    
        # Load state dict
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['module']  
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Checkpoint loaded from {path}: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        
        # Return additional state
        additional_state = {k: v for k, v in checkpoint.items() 
                          if k not in ['module', 'config']}
        return additional_state

    def load_pretrain_weights(self, path: str) -> None:
        """Load weights from a pretrain checkpoint when current mode is finetune.

        Skips layers that depend on state vs action-only differences:
          - action_expert.input_encoder.*
          - action_expert.decoder.*
        """
        if self.config.training_mode != 'finetune':
            raise ValueError("load_pretrain_weights should be called only in finetune mode")
        # Handle directory path (align with load_checkpoint style)
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
            path = str(checkpoint_file)

        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('module', checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if ('action_expert.input_encoder' in k or 'action_expert.decoder' in k):
                continue
            filtered[k] = v
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        logger.info(f"Loaded pretrain weights (filtered). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def training_step(
        self,
        first_frame: torch.Tensor,         # [B, C, H, W] - first frame
        video_frames: torch.Tensor,       # [B, num_frames, C, H, W] - target frames
        state: torch.Tensor = None,       # [B, state_dim] - robot state
        actions: torch.Tensor = None,     # [B, chunk_size, action_dim] - actions
        language_embeddings: Optional[List[torch.Tensor]] = None,  # Pre-encoded T5 embeddings for WAN
        vlm_inputs: Optional[List] = None,  # Complete VLM inputs from dataset
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        UniDiffuser training step with three modalities.
        
        Args:
            first_frame: First video frame for Teacher Forcing
            video_frames: Target video frames
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state
            actions: Target action sequence
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            return_dict: Whether to return detailed outputs
            
        Returns:
            Dictionary containing losses and metrics
        """
        B = video_frames.shape[0]  # batch size

        # 1. Video pipeline
        # Normalize/format,从[0, 1]映射到[-1, 1]
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, T_video, H, W]
        full_video = torch.cat([first_frame_norm, video_normalized], dim=2)  # [B, C, 1+T_video, H, W]

        # Encode video using VAE
        with torch.no_grad():
            clean_full_latent = self.video_model.encode_video(full_video.to(self.dtype))  # [B, C_latent, T_latent, H_latent, W_latent]
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))  # [B, C_latent, 1, H_latent, W_latent]

        # Flow-Matching noise mixture,B个均匀分布的随机整数标量，范围是[0, num_train_timesteps)，num_train_timesteps通常是1000
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))  # [B]
        # Scalar timesteps (0..num_train_timesteps) for time embedding
        # video_t_embed和timestep_id不同在于video_t_embed是经过scheduler的timesteps映射后的值，通常是一个连续的值，而timestep_id是离散的整数索引
        video_t_embed = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for noise mixture
        # sigma和video_t_embed是一一对应的，都是根据timestep_id从scheduler中取出的值。video_t_embed用于时间嵌入，而sigma用于控制视频latent和噪声的混合比例。随着timestep_id增加，通常video_t_embed会增加（具体取决于scheduler的设计），而sigma也会相应调整，使得训练过程中模型逐渐适应更高噪声水平的视频输入。
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)  # [B,1,1,1,1]
        video_noise = torch.randn_like(clean_full_latent, dtype=self.dtype)  # [B, C_latent, T_latent, H_latent, W_latent]
        # sigma=0时完全是clean_full_latent，sigma=1时完全是video_noise，中间值则是两者的线性混合
        noisy_video_latent = clean_full_latent * (1 - sigma) + video_noise * sigma  # [B, C_latent, T_latent, H_latent, W_latent]
        # Teacher Forcing on the first frame，只在[B, C_latent, T_latent，H_latent, W_latent]的T维度的第0帧位置强制使用condition_frame_latent，确保模型在训练时始终以第一帧的真实信息作为条件输入
        noisy_video_latent[:, :, 0:1] = condition_frame_latent
        # Flow-Matching target: noise - clean
        # noisy_video_latent对sigma求导
        video_target = video_noise - clean_full_latent  # [B, C_latent, T_latent, H_latent, W_latent]
        video_target[:, :, 0:1] = 0 # 因为第一帧没有噪声，所以对应的target应该是0

        # Latent to Tokens
        video_tokens = self.video_module.prepare_input(noisy_video_latent.to(self.dtype))  # [B, L_v, D_wan]

        # 2. Action pipeline 
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))  # [B]
        # Discrete timesteps for time embedding (0..num_train_timesteps)
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        # Sigma for action noise mixture
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)  # [B,1,1]
        action_noise = torch.randn_like(actions, dtype=self.dtype)  # [B, L_a, D_action_raw]
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action  # [B, L_a, D_action_raw]
        action_target = action_noise - actions  # [B, L_a, D_action_raw]

        # Encode Action Chunk with optional Registers
        if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
        else:
            registers = None
        if self.config.training_mode == 'pretrain':
            action_tokens = self.action_expert.input_encoder(None, noisy_actions, registers)  # [B, L_a(+reg), D_action_hidden]
        else:
            state_tokens = state.unsqueeze(1).to(self.dtype)  # [B, 1, state_dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, noisy_actions, registers)  # [B, 1+L_a(+reg), D_action_hidden]

        und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, seq_len, und_dim]

        # Time embeddings
        # Use scheduler-provided timesteps (0..num_train_timesteps) for WAN/action time embeddings
        video_head_time_emb, video_adaln_params  = self.video_module.get_time_embedding(video_t_embed, video_tokens.shape[1])
        action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_embed, action_tokens.shape[1])

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. MoT forward
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            # Process through 30 layers - modality-grouped execution
            for layer_idx in range(self.config.num_layers):
                # Compute AdaLN modulation once per layer using pre-computed parameters
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                
                # Trimodal MoT: WAN + Action + Understanding Expert joint attention
                video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                    video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                    self.action_expert.blocks[layer_idx],
                    und_tokens, self.und_expert.blocks[layer_idx]
                )

                # WAN cross
                video_tokens = self.video_module.process_cross_attention(video_tokens, video_adaln_params, layer_idx, processed_t5_context)

                # FFNs: WAN, Action, Understanding (each processes their own FFN)
                video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)
                
        
            # 4. Heads + Losses
            video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [B, C_latent, T_latent, H_latent, W_latent]
            action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)  # [B, L_decoded, D_action_raw]
            up_len = action_pred_full.shape[1] - self.action_expert.config.num_registers
            # Slice predicted actions depending on mode
            if self.config.training_mode == 'pretrain':
                action_pred = action_pred_full[:, :up_len, :]  # [B, L_a, D_action_raw]
            else: # 刨除index为0的状态token
                action_pred = action_pred_full[:, 1:up_len, :]  # [B, L_a, D_action_raw]

            # Video loss (mask the first frame)
            video_pred_masked = video_pred.clone()  # [B, C_latent, T_latent, H_latent, W_latent]
            video_pred_masked[:, :, 0:1] = 0 # 把video_pred的第一帧置零，和video_target的第一帧一致，确保损失计算时第一帧不贡献误差
            video_loss = torch.nn.functional.mse_loss(video_pred_masked, video_target, reduction='mean')
        
            # Action loss
            action_loss = torch.nn.functional.mse_loss(action_pred, action_target, reduction='mean')

        total_loss = (
            self.config.video_loss_weight * video_loss +
            self.config.action_loss_weight * action_loss
        )
        
        if return_dict:
            return {
                'total_loss': total_loss,
                'video_loss': video_loss,
                'action_loss': action_loss,
                'video_timestep_mean': sigma.float().mean().item(),
                'action_timestep_mean': sigma_action.float().mean().item(),
            }

    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame `[B, C=3, H, W]`, value range `[0,1]`
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]  # batch size，真实世界示例通常为 1

        # device和dtype consistency: 确保所有输入都在正确的设备和数据类型上，避免运行时错误
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]  # List[[L_t5_i, D_t5]]
        state = state.to(self.device).to(self.dtype)          # [B, state_dim]
        first_frame = first_frame.to(self.device).to(self.dtype)  # [B, 3, H, W]

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, H, W] ---> [B, C, T=1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C_latent, 1, H_latent, W_latent]

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        # C_latent: VAE latent 通道数
        # f_latent: 条件帧 latent 帧数，单帧条件下通常为 1
        # H_latent / W_latent: latent 空间分辨率
        # 这里的 `// 4` 来自 WAN VAE 的时间压缩倍率：
        # - 在 `bak/wan/configs/wan_ti2v_5B.py` 里可见 `vae_stride = (4, 16, 16)`
        #   其中第 1 个维度 4 表示“时间维压缩 4 倍”
        # - 在 `bak/wan/textimage2video.py` 里，官方计算 latent 时间长度的公式是
        #   `(F - 1) // self.vae_stride[0] + 1`
        #   即像素空间视频帧数 `F` 会按 `vae_stride[0]` 映射到 latent 帧数
        # 所以这里:
        # - `1` 表示条件首帧的 latent
        # - `self.config.num_video_frames // 4` 表示未来 `num_video_frames`
        #   在 latent 空间里对应的帧数
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)  # [B, C_latent, T_latent, H_latent, W_latent]
        video_latent[:, :, 0:1] = condition_frame_latent # 将latent空间的第一帧初始化为条件帧的latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)  # [B, L_a, D_action_raw]
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)  # [B, L_a, D_action_raw]

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, L_u, D_und]

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)  # [B, L_t5_fixed, D_wan]

        # 3. Denoising loop: from noise (t=1) to clean (t=0)
        # 从噪声（t=1）到干净（t=0）的时间序列
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)  # [num_inference_steps+1]
        for i in range(num_inference_steps):
            # Timesteps
            t = timesteps[i]           # 当前时刻，标量
            t_next = timesteps[i + 1]  # 下一时刻，标量
            dt = t_next - t            # Euler 步长，标量
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)   # [B]
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)  # [B]

            # Tokens with Registers
            # 将视频的潜变量转换为WAN模型的token
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))  # [B, L_v, D_wan]

            # 将状态和动作潜变量转换为action expert的token
            state_tokens = state.unsqueeze(1).to(self.dtype)  # [B, 1, state_dim]

            # Expand registers for batch
            registers = self.action_expert.registers.expand(B, -1, -1)  # [B, num_registers, dim]
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)  # [B, 1+L_a+num_registers, D_action_hidden]

            # Note: Understanding tokens already extracted before the loop, will be updated in joint attention
            und_tokens = self.und_module.extract_und_features(vlm_inputs)  # [B, L_u, D_und]

            
            # Trimodal MoT forward - joint denoising for WAN, Action, Understanding
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(action_t_scaled, action_tokens.shape[1])

                # Process through all layers - trimodal denoising of WAN, Action, Understanding
                for layer_idx in range(self.config.num_layers):
                    # Compute AdaLN modulation using pre-computed parameters
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    
                    # Trimodal joint attention: WAN + Action + Understanding
                    # MoT核心：WAN，Action，Understanding三模态在每层的联合注意力机制
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )

                    # WAN cross-attention with T5 embeddings 
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )

                    # FFNs: WAN, Action, Understanding
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Heads (velocities)
                # 从token空间到latent空间的解码，得到视频和动作的velocity
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [B, C_latent, T_latent, H_latent, W_latent]
                # Use decoder with all tokens (including registers)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)  # [B, 1+L_a+num_registers, D_action_raw]
                # Extract middle action chunk (skip first state token and last register tokens)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]  # [B, L_a, D_action_raw]

                # Euler integration
                video_latent = video_latent + video_velocity * dt    # [B, C_latent, T_latent, H_latent, W_latent]
                action_latent = action_latent + action_velocity * dt # [B, L_a, D_action_raw]

                # Teacher Forcing
                video_latent[:, :, 0:1] = condition_frame_latent

        # 4. Decode outputs
        with torch.no_grad():
            # 从latent空间到像素空间的解码，得到预测的视频帧
            decoded_frames = self.video_model.decode_video(video_latent)  # 通常为 [B, 3, 1+T_video, H, W]
            predicted_frames = decoded_frames[:, :, 1:]  # [B, 3, T_video, H, W]，去掉条件帧
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] -> [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()  # [B, 3, T_video, H, W]
        # action latent空间和action实际空间一样，所以直接输出即可
        predicted_actions = action_latent.float()  # [B, action_chunk_size, action_dim]

        return predicted_frames, predicted_actions

    # Alternative inference (DPM++ solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None,
        solver: Optional[str] = None,
        shift: Optional[float] = None,
        seed: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction with dpm++ solver.
        
        Args:
            first_frame: Initial frame [1, C, H, W] - batch size must be 1 for inference
            state: Initial robot state [1, state_dim]
            num_inference_steps: Number of denoising steps (default: 50)
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            vlm_inputs: VLM inputs for understanding expert
            solver: Solver type ("dpm++"), defaults to config.inference_solver
            shift: Noise schedule shift, defaults to config.inference_shift
            seed: Random seed for reproducible generation (-1 for random)
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        # Move inputs to device
        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)
        
        # Use config defaults if not specified
        if solver is None:
            solver = self.config.inference_solver
        if shift is None:
            shift = self.config.inference_shift

        # Set random seed if specified
        if seed >= 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode condition frame and initialize latents
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [1, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [1, 48, 1, H', W']

        # Initialize video latent with noise - squeeze batch dimension for WAN format
        _, C_latent, _, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn(
            (C_latent, num_total_latent_frames, H_latent, W_latent), 
            device=self.device, 
            dtype=torch.float32,  # Use float32 for sampling numerical stability
            generator=generator
        )
        # Set first frame as condition (teacher forcing)
        video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()
        
        # Initialize action latent with noise
        action_latent = torch.randn(
            (1, self.config.action_chunk_size, self.config.action_dim), 
            device=self.device, 
            dtype=torch.float32,
            generator=generator
        )

        # 2. Prepare understanding features and T5 context (compute once, reuse for all steps)
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Setup flow-matching schedulers (separate for video and action due to different tensor shapes)
        if solver == "dpm++":
            # Video scheduler
            video_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,  # Base shift is 1.0
                use_dynamic_shifting=False
            )
            # Action scheduler (independent instance)
            action_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                shift=1.0,
                use_dynamic_shifting=False
            )
            # Get custom sigmas with shift parameter
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                video_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
            # Set same timesteps for action scheduler
            _, _ = retrieve_timesteps(
                action_scheduler,
                device=self.device,
                sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(f"Solver '{solver}' not implemented. Currently only 'dpm++' is supported.")

        # 4. Denoising loop with flow-matching solver
        with torch.no_grad():
            for step_idx, t in enumerate(timesteps):
                # Prepare model inputs (add batch dimension back)
                video_latent_input = video_latent.unsqueeze(0).to(self.dtype)  # [1, 48, T, H, W]
                action_latent_input = action_latent.to(self.dtype)  # [1, chunk_size, action_dim]
                
                # Prepare tokens
                video_tokens = self.video_module.prepare_input(video_latent_input)
                state_tokens = state.unsqueeze(1)
                registers = self.action_expert.registers.expand(1, -1, -1)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent_input, registers)
                und_tokens = self.und_module.extract_und_features(vlm_inputs)
                
                # Model forward pass
                with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                    # Time embeddings (t is in [0, 1000] from scheduler)
                    video_t_scaled = t.expand(1).to(self.dtype)
                    action_t_scaled = t.expand(1).to(self.dtype)
                    video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                        video_t_scaled, video_tokens.shape[1]
                    )
                    action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                        action_t_scaled, action_tokens.shape[1]
                    )

                    # Process through all layers - trimodal joint denoising
                    for layer_idx in range(self.config.num_layers):
                        video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                        action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                        
                        # Trimodal joint attention
                        video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                            video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                            self.action_expert.blocks[layer_idx],
                            und_tokens, self.und_expert.blocks[layer_idx]
                        )

                        # WAN cross-attention with T5
                        video_tokens = self.video_module.process_cross_attention(
                            video_tokens, video_adaln_params, layer_idx, processed_t5_context
                        )

                        # FFNs for each modality
                        video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                        action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                        und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                    # Prediction heads (predict velocity for flow-matching)
                    video_pred = self.video_module.apply_output_head(video_tokens, video_head_time_emb)  # [1, 48, T, H, W]
                    action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                    action_pred = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]  # [1, chunk_size, action_dim]

                # Update latents using separate schedulers (video and action have different tensor shapes)
                # Video: squeeze batch dim, call video_scheduler, squeeze back
                video_latent = video_scheduler.step(
                    video_pred.squeeze(0).unsqueeze(0),  # Add dummy batch dim for scheduler
                    t,
                    video_latent.unsqueeze(0),  # Add dummy batch dim for scheduler
                    return_dict=False,
                    generator=generator
                )[0].squeeze(0)  # Remove dummy batch dim
                
                # Action: directly use 3D tensor [1, chunk_size, action_dim] with action_scheduler
                # DPM-Solver doesn't require specific dimensions, just consistency
                action_latent = action_scheduler.step(
                    action_pred,  # [1, chunk_size, action_dim]
                    t,
                    action_latent,  # [1, chunk_size, action_dim]
                    return_dict=False,
                    generator=generator
                )[0]
                
                # Teacher forcing: keep first frame as condition
                video_latent[:, 0:1] = condition_frame_latent.squeeze(0).float()

        # 5. Decode final outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent.unsqueeze(0).to(self.dtype))  # Add batch dim back
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [1, chunk_size, action_dim]

        return predicted_frames, predicted_actions
    '''

    # Alternative inference (UniPC solver)
    '''
    def inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint inference for video and action prediction.
        
        Args:
            first_frame: Initial frame `[B, C=3, H, W]`, value range `[0,1]`
            texts: Text instructions for VLM
            images: Optional images for VLM
            state: Initial robot state [B, state_dim]
            num_inference_steps: Number of denoising steps
            language_embeddings: Pre-encoded T5 embeddings for WAN model
            
        Returns:
            Tuple of (predicted_frames, predicted_actions)
        """
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        if self.config.training_mode != 'pretrain':
            state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Video/Action latents init
        # Condition frame encode
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)   # [0,1] -> [-1,1], [B, C, 1, H, W]
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))   # [B, C', 1, H', W']

        # Init video/action latents
        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features and T5 context
        # Extract understanding features from VLM
        und_tokens = self.und_module.extract_und_features(vlm_inputs)

        # T5 preprocess
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Denoising loop: use FlowUniPCMultistepScheduler for video and action latents
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        # Use a separate scheduler instance for the action branch to avoid shared internal state
        action_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)
        action_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=1.0)
        timesteps = scheduler.timesteps  # int64 on device

        for t in timesteps:
            # Tokens (with optional registers)
            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            if self.action_expert.config.num_registers > 0 and self.action_expert.registers is not None:
                registers = self.action_expert.registers.expand(B, -1, -1)
            else:
                registers = None
            if self.config.training_mode == 'pretrain':
                action_tokens = self.action_expert.input_encoder(None, action_latent, registers)
            else:
                state_tokens = state.unsqueeze(1).to(self.dtype)
                action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            # Re-extract understanding features per step (keeps alignment with current pipeline)
            und_tokens = self.und_module.extract_und_features(vlm_inputs)
            
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                # Time embeddings: use the current discrete t (0..num_train_timesteps)
                t_scalar = t.to(self.dtype).repeat(B)
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(t_scalar, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(t_scalar, action_tokens.shape[1])

                # Layer stack for joint denoising
                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens, video_adaln_modulation, action_adaln_modulation, layer_idx, 
                        self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx]
                    )
                    # WAN cross
                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context
                    )
                    video_tokens = self.video_module.process_ffn(video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # Predict velocities (video and action) and take scheduler steps
                video_velocity = self.video_module.apply_output_head(video_tokens, video_head_time_emb)
                action_velocity_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                up_len = action_velocity_full.shape[1] - self.action_expert.config.num_registers
                if self.config.training_mode == 'pretrain':
                    action_velocity = action_velocity_full[:, :up_len, :]
                else:
                    action_velocity = action_velocity_full[:, 1:up_len, :]

                # Scheduler steps
                video_latent = scheduler.step(model_output=video_velocity, timestep=t, sample=video_latent, return_dict=False)[0]
                # Teacher Forcing on the first frame (video)
                video_latent[:, :, 0:1] = condition_frame_latent
                action_latent = action_scheduler.step(model_output=action_velocity, timestep=t, sample=action_latent, return_dict=False)[0]

        # 4. Decode outputs
        with torch.no_grad():
            decoded_frames = self.video_model.decode_video(video_latent)
            predicted_frames = decoded_frames[:, :, 1:]  # Skip first frame (condition)
            predicted_frames = (predicted_frames + 1.0) / 2.0  # [-1,1] to [0,1]
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()

        predicted_actions = action_latent.float()  # [B, action_chunk_size, action_dim]

        return predicted_frames, predicted_actions
    '''


def test_motus():
    """Test the complete model."""
    print("Testing Motus...")

    config = MotusConfig()

    try:
        model = Motus(config)
        print("Model created successfully")

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")

    except Exception as e:
        print(f"Model creation failed: {e}")
        print("This is expected without actual pretrained weights")

if __name__ == "__main__":
    test_motus()
