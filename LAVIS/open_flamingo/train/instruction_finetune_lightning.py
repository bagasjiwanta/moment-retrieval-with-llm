import lightning as L
import argparse
from datetime import datetime
import os
from omegaconf import OmegaConf
import torch
from typing import Optional, List, Tuple
import wandb
import functools
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from open_flamingo import create_model_and_transforms, SUPPORTED_MODEL_FAMILIES
from open_flamingo.train.distributed import (
    init_distributed_device,
    world_info_from_env,
    get_fsdp_config,
    get_fsdp_checkpoint_config,
)
from open_flamingo.train.sft_data_utils import make_supervised_data_module
from open_flamingo.train.train_utils import (
    finetune_one_epoch,
    random_seed,
    find_most_recent_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from open_flamingo.train.losses import (
    SUPPORTED_LOSSES,
    get_loss_fn,
)
from open_flamingo.train.args import parse_args
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


class MyLightningModule(L.LightningModule):
    def __init__(
        self,
        # model kwargs
        clip_vision_encoder_path: str,
        clip_vision_encoder_pretrained: str,
        lm_path: str,
        tokenizer_path: Optional[str],
        model_family: str,
        pretrained_vision_tokenizer: str,
        offline: bool,
        gradient_checkpointing: bool,
        verbose:bool,
        anyres_grids: List[Tuple[int]],
        use_flash_attention_2: bool,
        image_aspect_ratio: str,
        num_vision_tokens: int,
        anyres_patch_sampling: bool,
        run_name: str 
    ):
        self.save_hyperparameters()
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path,
            clip_vision_encoder_pretrained,
            lm_path,
            tokenizer_path if tokenizer_path else lm_path,
            model_family,
            pretrained_vision_tokenizer,
            offline,
            gradient_checkpointing, 
            verbose,
            anyres_grids,
            use_flash_attention_2,
            image_aspect_ratio,
            num_vision_tokens,
            anyres_patch_sampling
        )

