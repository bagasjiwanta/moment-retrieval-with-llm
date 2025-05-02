# ALL THE ARGUMENTS HERE BCS ALL THIS UNTYPED ARGS EVERYWHERE 
# loss = get_loss_fn(args.loss)  bro what's this how do u read n debug
# _, _, checkpoint = load_checkpoint(args, model, pretrained=True)  also what's this?
# in the code, the args are never typed but it's super crucial and annoying to keep track


from dataclasses import dataclass, field
from typing import List, Tuple, Optional

def parse_tuple_list(val: str) -> List[Tuple[int, int]]:
    return [tuple(map(int, x.strip("()").split(","))) for x in val.split(",") if x]

@dataclass
class Args:
    # Model configuration
    model_family: str = "xgenmm_v1"
    vision_encoder_path: str = "google/siglip-so400m-patch14-384"
    vision_encoder_pretrained: str = "google"
    lm_path: str = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer_path: str = "microsoft/Phi-3-mini-4k-instruct"
    cross_attn_every_n_layers: int = 1
    num_vision_tokens: int = 128
    pretrained: Optional[str] = "/workspace/LAVIS/base_model_weight/xgen-mm-phi3-mini-base-r-v1.5.pt"
    pretrained_vision_tokenizer: Optional[str] = None

    # Training args
    loss: str = "supervised_finetune"
    run_name: str = "openflamingo3B"
    resume_from_checkpoint: Optional[str] = None
    delete_previous_checkpoint: bool = False
    no_save_optim_state: bool = False
    gradient_accumulation_steps: int = 1
    seed: int = 42
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    warmup_steps: int = 5000
    weight_decay: float = 0.1
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    num_epochs: int = 1
    offline: bool = False
    logging_steps: int = 100
    checkpoint_steps: int = 5000

    # Data args
    data_path: str = "/export/home/LLaVA/playground/data/llava_v1_5_mix665k_ocr_tagged_vqa_placeholder.json"
    batch_size: int = 8
    workers: int = 1
    data_sampler_group_by_length: bool = False
    is_multimodal: bool = True
    mm_use_im_start_end: bool = False
    conv_template_name: Optional[str] = None
    image_aspect_ratio: str = "pad"
    anyres_patch_sampling: bool = False
    anyres_grids: List[Tuple[int, int]] = field(default_factory=lambda: [(1,2), (2,1), (2,2), (3,1), (1,3)])

    # Distributed training args
    dist_url: str = "env://"
    dist_backend: str = "nccl"
    horovod: bool = False
    no_set_device_rank: bool = False
    local_rank: int = 0

    # FSDP args
    fsdp: bool = False
    fsdp_sharding_strategy: str = "full"

    # Wandb args
    report_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    save_checkpoints_to_wandb: bool = False
    dryrun: bool = False

    # Extra
    use_flash_attention_2: bool = False
    unfreeze_vision_encoder: bool = False
    vision_encoder_precision: str = "fp32"
    cpu_offload_gradients: bool = False
