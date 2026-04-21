#!/usr/bin/env python3
"""
Real-World Motus Inference Example (No Environment Required)

This script demonstrates how to run Motus inference on a single image without any robot environment.
It supports two modes:
1. With T5: encode instruction text on the fly
2. Without T5: use pre-encoded T5 embeddings

Example usage (similar to RDT2 style):
```python
import torch
import yaml
from PIL import Image
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.motus import Motus, MotusConfig
from transformers import AutoProcessor
from wan.modules.t5 import T5EncoderModel

# Load config
with open("inference/real_world/Motus/utils/aloha_agilex_2.yml", "r") as f:
    config = yaml.safe_load(f)

# Create model
device = "cuda:0"
model_config = MotusConfig(
    wan_checkpoint_path=config['model']['wan']['checkpoint_path'],
    vae_path=config['model']['wan']['vae_path'],
    wan_config_path=config['model']['wan']['config_path'],
    video_precision=config['model']['wan']['precision'],
    vlm_checkpoint_path=config['model']['vlm']['checkpoint_path'],
    # ... other configs from yaml
    load_pretrained_backbones=False,  # Load from checkpoint instead
)
model = Motus(model_config).to(device).eval()

# Load checkpoint
model.load_checkpoint("/path/to/checkpoint_step_xxxxx", strict=False)

# Prepare inputs
first_frame = Image.open("/path/to/image.png").convert("RGB")
first_frame_tensor = torch.from_numpy(np.array(first_frame.resize((320, 384)))).permute(2,0,1).unsqueeze(0).float() / 255.0
state = torch.zeros((1, config['common']['state_dim']), dtype=torch.bfloat16, device=device)

# Build VLM inputs
processor = AutoProcessor.from_pretrained(config['model']['vlm']['checkpoint_path'], trust_remote_code=True)
vlm_inputs = processor(text=["Pick up the cube."], images=[first_frame], return_tensors='pt')
vlm_inputs = {k: v.to(device) for k, v in vlm_inputs.items()}

# Option 1: Encode instruction with T5
t5_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path="/path/to/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    tokenizer_path="/path/to/Wan2.2-TI2V-5B/google/umt5-xxl",
)
language_embeddings = t5_encoder(["Pick up the cube."], device)

# Option 2: Use pre-encoded T5 embeddings
# language_embeddings = torch.load("/path/to/preencoded_t5.pt", map_location=device)
# if isinstance(language_embeddings, torch.Tensor):
#     language_embeddings = [language_embeddings]

# Run inference
with torch.no_grad():
    predicted_frames, predicted_actions = model.inference_step(
        first_frame=first_frame_tensor.to(device),
        state=state,
        num_inference_steps=config['model']['inference']['num_inference_timesteps'],
        language_embeddings=language_embeddings,
        vlm_inputs=[vlm_inputs],
    )

# predicted_frames: torch.Tensor of shape (B, T, C, H, W) or (B, C, T, H, W)
# predicted_actions: torch.Tensor of shape (B, action_chunk_size, action_dim)
#   - action_chunk_size = num_video_frames * video_action_freq_ratio
#   - action_dim: robot action dimension (e.g., 14 for single arm)
```

This file also provides a command-line interface for convenience.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from PIL import Image
import yaml

# Add project root to import model
PROJ_ROOT = str(Path(__file__).resolve().parents[3])
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.motus import Motus, MotusConfig
from transformers import AutoProcessor
from wan.modules.t5 import T5EncoderModel


def load_yaml_config(path: str) -> Dict[str, Any]:
    """读取 YAML 配置。

    返回:
        Dict[str, Any]:
            配置树本身，不涉及张量维度。
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_image_as_tensor(image_path: str, size_hw: tuple[int, int]) -> torch.Tensor:
    """把单张 RGB 图片读成模型使用的归一化张量。

    参数:
        image_path:
            输入图片路径。
        size_hw:
            目标尺寸 `(H, W)`。

    返回:
        tensor: `torch.Tensor`，形状 `[1, C, H, W]`
            - `1`: batch size，示例脚本固定为单样本
            - `C=3`: RGB 三通道
            - `H/W`: 由 `size_hw` 指定
            - 值域: `[0, 1]`
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size_hw[1], size_hw[0]), Image.BICUBIC)  # (W,H)
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, C]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    return tensor


def build_vlm_inputs(processor, instruction: str, image: Image.Image, device: torch.device) -> Dict[str, torch.Tensor]:
    """构造 Qwen3-VL 所需的多模态输入。

    返回字典中的主要字段维度:
        - `input_ids`: `[1, L]`
        - `attention_mask`: `[1, L]`
        - `pixel_values`: `[N_img_patch, C_patch]` 或模型内部约定格式
          这里具体 patch 展平后的形状由 processor 决定，因此只保证它是
          Qwen3-VL 可直接消费的视觉张量。
        - `image_grid_thw`: `[1, 3]`
          分别表示视觉 token 对应的 `(T, H, W)` 网格；单图场景下 `T=1`。
    """
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image', 'image': image},
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    encoded = processor(text=[text], images=[image], return_tensors='pt')
    vlm_inputs = {
        'input_ids': encoded['input_ids'].to(device),              # [1, L]
        'attention_mask': encoded['attention_mask'].to(device),    # [1, L]
        'pixel_values': encoded['pixel_values'].to(device),        # Qwen3-VL 视觉输入张量
        'image_grid_thw': encoded.get('image_grid_thw', None)
    }
    if vlm_inputs['image_grid_thw'] is not None:
        vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(device)  # [1, 3]
    return vlm_inputs


def save_frame_grid(condition_frame: torch.Tensor, predicted_frames: torch.Tensor, save_path: str) -> None:
    """把条件帧和预测帧横向拼接保存，便于肉眼检查。

    参数维度:
        - `condition_frame`: `[C, H, W]`
        - `predicted_frames`: `[T, C, H, W]`
          其中 `T=num_video_frames`
    """
    cf = (condition_frame.detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    frames = []
    T = predicted_frames.shape[0]
    for i in range(T):
        f = (predicted_frames[i].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        frames.append(f)
    all_frames = [cf] + frames
    grid = np.concatenate(all_frames, axis=1)
    Image.fromarray(grid).save(save_path)


def create_motus_from_yaml(config_dict: Dict[str, Any], device: torch.device) -> Motus:
    """根据 real-world YAML 配置构造 Motus 模型。

    这里主要做两件事:
        1. 从 YAML 中把 WAN / VLM / Action / Und 模块的超参数抽出来
        2. 组装成 `MotusConfig`，再实例化 `Motus`

    注意:
        本函数只负责“建图”，不负责加载训练权重。
    """
    common = config_dict['common']
    model_cfg = config_dict['model']
    mc = MotusConfig(
        wan_checkpoint_path=model_cfg['wan']['checkpoint_path'],
        vae_path=model_cfg['wan']['vae_path'],
        wan_config_path=model_cfg['wan']['config_path'],
        video_precision=model_cfg['wan']['precision'],
        vlm_checkpoint_path=model_cfg['vlm']['checkpoint_path'],
        und_expert_hidden_size=model_cfg.get('und_expert', {}).get('hidden_size', 512),
        und_expert_ffn_dim_multiplier=model_cfg.get('und_expert', {}).get('ffn_dim_multiplier', 4),
        und_expert_norm_eps=model_cfg.get('und_expert', {}).get('norm_eps', 1e-5),
        vlm_adapter_input_dim=model_cfg.get('und_expert', {}).get('vlm', {}).get('input_dim', 2048),
        vlm_adapter_projector_type=model_cfg.get('und_expert', {}).get('vlm', {}).get('projector_type', "mlp3x_silu"),
        num_layers=30,
        action_state_dim=common['state_dim'],
        action_dim=common['action_dim'],
        action_expert_dim=model_cfg['action_expert']['hidden_size'],
        action_expert_ffn_dim_multiplier=model_cfg['action_expert']['ffn_dim_multiplier'],
        action_expert_norm_eps=model_cfg['action_expert'].get('norm_eps', 1e-6),
        global_downsample_rate=common['global_downsample_rate'],
        video_action_freq_ratio=common['video_action_freq_ratio'],
        num_video_frames=common['num_video_frames'],
        video_height=common['video_height'],
        video_width=common['video_width'],
        batch_size=1,
        video_loss_weight=model_cfg['loss_weights']['video_loss_weight'],
        action_loss_weight=model_cfg['loss_weights']['action_loss_weight'],
        training_mode='finetune',
        load_pretrained_backbones=False,  # we will load from checkpoint
    )
    model = Motus(mc).to(device)
    return model


def load_checkpoint_into_model(model: Motus, ckpt_path: str) -> None:
    try:
        model.load_checkpoint(ckpt_path, strict=False)
        print(f"Loaded Motus checkpoint from {ckpt_path}")
    except Exception as e:
        print(f"WARNING: failed to load checkpoint: {e}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Real-World Motus inference sample (no env)")
    parser.add_argument("--model_config", required=True, help="Path to real-world YAML (e.g., inference/real_world/Motus/utils/aloha_agilex_2.yml)")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoint directory (contains mp_rank_00_model_states.pt or state dir)")
    parser.add_argument("--wan_path", required=True, help="Base path to WAN models (to find T5 and VAE)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--instruction", required=True, help="Instruction text")
    parser.add_argument("--output", default="inference_result.png", help="Where to save predicted frames grid")
    parser.add_argument("--use_t5", action="store_true", help="Load T5 and encode instruction on the fly")
    parser.add_argument("--t5_embeds", default=None, help="Path to pre-encoded T5 embeddings (.pt) when not using --use_t5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 推理设备

    # Load config
    # 加载yaml配置，包含了motus初始化需要的配置文件的内容
    cfg = load_yaml_config(args.model_config)

    # Create model
    # 因为load_pretrained_backbones=False，所以走的分支都是from_config,通过yaml配置创建Motus模型实例
    model = create_motus_from_yaml(cfg, device)
    model.eval()
    # 加载这次具体训练出来的 Motus 权重
    # 即使yaml里指定了WAN/VLM的checkpoint_path，这里也不直接加载它们的权重，而是通过load_checkpoint加载整个Motus模型的训练权重（其中包含了WAN/VLM的权重），以确保视觉编码器和语言编码器等子模块都正确加载到训练好的状态。
    load_checkpoint_into_model(model, args.ckpt_dir)

    # Prepare inputs
    H, W = cfg['common']['video_height'], cfg['common']['video_width']  # 目标视频尺寸
    # 将输入图像转换成模型需要的张量格式，并移动到正确的设备上
    first_frame = load_image_as_tensor(args.image, (H, W)).to(device)  # [1, 3, H, W]
    state_dim = int(cfg['common']['state_dim'])
    state = torch.zeros((1, state_dim), dtype=torch.bfloat16, device=device)  # [1, state_dim]
    # 真实机器人示例里这里通常应换成当前观测到的机器人状态；
    # 本脚本没有环境，因此用全 0 状态占位。

    # Build VLM inputs
    vlm_ckpt = cfg['model']['vlm']['checkpoint_path']
    # `vlm_ckpt` 通常指向一个 Hugging Face 风格的 Qwen3-VL 目录，例如:
    #   Qwen3-VL-2B-Instruct/
    #     |- config.json
    #     |- model.safetensors
    #     |- generation_config.json
    #     |- preprocessor_config.json
    #     |- video_preprocessor_config.json
    #     |- tokenizer_config.json
    #     |- tokenizer.json
    #     |- chat_template.json
    #     |- vocab.json
    #     |- merges.txt
    #
    # 这些文件会被两类组件“分别”读取，但职责不同，不是重复加载同一份内容：
    #
    # 1. Motus 里的 VLM（`model.vlm_model`）主要关心:
    #    - config.json:
    #        用来构造 Qwen3-VL 的网络结构定义，例如 hidden size、层数、注意力头数、
    #        vision/text 子模块配置等。
    #    - model.safetensors:
    #        这是 Qwen3-VL 的参数权重文件。
    #        但当前示例里 `load_pretrained_backbones=False`，所以初始化时只读取 config
    #        来“搭骨架”，不会直接从这里加载最终参数；最终参数是后面通过
    #        `load_checkpoint_into_model(model, args.ckpt_dir)` 从 Motus checkpoint 恢复的。
    #
    # 2. Processor（`AutoProcessor`）主要关心:
    #    - tokenizer.json / tokenizer_config.json / vocab.json / merges.txt:
    #        负责把文本变成 `input_ids`
    #    - preprocessor_config.json / video_preprocessor_config.json:
    #        负责图像/视频 resize、normalize、patch/grid 等预处理规则
    #    - chat_template.json:
    #        负责把多模态消息拼成 Qwen3-VL 期望的对话模板
    #
    # 所以：
    # - VLM 负责“前向计算”
    # - processor 负责“把原始文本/图像整理成 VLM 能吃的输入”
    # 两者都用同一个目录，但读取的文件类型不同，因此不会互相加载错权重。
    processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=True)

    # 将输入的图像转成模型所需的张量格式
    first_frame_pil = Image.open(args.image).convert("RGB").resize((W, H), Image.BICUBIC)
    
    # 将文本指令和图像一起处理成VLM输入格式，包含input_ids, attention_mask, pixel_values等，并移动到设备上
    vlm_inputs = build_vlm_inputs(processor, args.instruction, first_frame_pil, device)

    # Build T5 embeddings
    if args.use_t5:
        t5_ckpt = os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'models_t5_umt5-xxl-enc-bf16.pth')
        t5_tokenizer = os.path.join(args.wan_path, 'Wan2.2-TI2V-5B', 'google/umt5-xxl')
        t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=str(device),
            checkpoint_path=t5_ckpt,
            tokenizer_path=t5_tokenizer,
        )
        t5_out = t5([args.instruction], device=str(device))  # 常见返回: [1, L_t5, D_t5]
        if isinstance(t5_out, torch.Tensor):
            language_embeddings: List[torch.Tensor] = [t5_out.squeeze(0)]  # 每个元素: [L_t5, D_t5]
        else:
            language_embeddings = t5_out
    else: # 如果不使用T5编码指令，则需要提供预编码的T5嵌入文件路径，通过命令行参数传入
        if args.t5_embeds is None:
            raise ValueError("Please provide --t5_embeds when not using --use_t5")
        loaded = torch.load(args.t5_embeds, map_location=device)
        # 允许两种格式:
        # 1. 单个 Tensor: [L_t5, D_t5]
        # 2. Tensor 列表: List[[L_t5_i, D_t5]]
        if isinstance(loaded, torch.Tensor):
            language_embeddings = [loaded.to(device)]
        elif isinstance(loaded, list):
            language_embeddings = [t.to(device) for t in loaded]
        else:
            raise ValueError("Unsupported t5_embeds format, expected Tensor or List[Tensor]")

    # Inference
    with torch.no_grad():
        predicted_frames, predicted_actions = model.inference_step(
            first_frame=first_frame,  # [1, 3, H, W]
            state=state,              # [1, state_dim]
            num_inference_steps=cfg['model']['inference']['num_inference_timesteps'],
            language_embeddings=language_embeddings,  # List[[L_t5, D_t5]]
            vlm_inputs=[vlm_inputs],                  # List[Dict[str, Tensor]]，batch size=1
        )
    # language_embeddings和vlm_inputs都涉及了args.instruction，但它们分别是给WAN和VLM使用的不同格式的输入：
    # language_embeddings 是给 WAN 的文本条件
    # vlm_inputs 是给 VLM 的多模态输入

    # Save frames grid
    # Convert predicted_frames to [T,C,H,W]
    if predicted_frames.dim() == 5 and predicted_frames.shape[1] != 3:
        frames_vis = predicted_frames.squeeze(0) # [B=1, T, C, H, W] ---> [T, C, H, W]
    else:
        frames_vis = predicted_frames.permute(0, 2, 1, 3, 4).squeeze(0) # [B=1, C, T, H, W] ---> [B=1, T, C, H, W] ---> [T, C, H, W]
    save_frame_grid(first_frame.squeeze(0), frames_vis, args.output)
    print(f"Saved predicted frames grid to {args.output}")

    # Print actions
    print("Predicted actions shape:", tuple(predicted_actions.shape))  # 典型: [1, action_chunk_size, action_dim]
    print("First 3 actions:\n", predicted_actions.squeeze(0)[:3].float().cpu().numpy())  # [3, action_dim]


if __name__ == "__main__":
    main()
