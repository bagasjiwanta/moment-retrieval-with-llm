# ALL THE ARGUMENTS HERE BCS ALL THIS UNTYPED ARGS EVERYWHERE 
# loss = get_loss_fn(args.loss)  bro what's this how do u read n debug
# _, _, checkpoint = load_checkpoint(args, model, pretrained=True)  also what's this?
# in the code, the args are never typed but it's super crucial and annoying to keep track


from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import argparse


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


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    # model configuration args
    parser.add_argument("--model_family", default="xgenmm_v1", type=str, choices=['xgenmm_v1'])
    parser.add_argument("--vision_encoder_path", default="ViT-SO400M-14-SigLIP-384", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="webli", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument("--lora", default=False, action="store_true")
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--num_vision_tokens",
        type=int,
        default=64,
        help="number of query tokens used for resampling vision features.",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained weights for fine-tuning.")
    parser.add_argument(
        "--pretrained_vision_tokenizer", type=str, default=None, help="pretrained vl connector for fine-tuning."
    )

    # training args
    parser.add_argument("--loss", type=str, choices=['supervised_finetune'], default="supervised_finetune")
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default.",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument(
        "--no_save_optim_state",
        action="store_true",
        help="do not save optimizer states when saving checkpoints",
    )

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bf16",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples specified by train_num_samples, not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=50, help="log loss every n steps")
    parser.add_argument("--checkpoint_steps", type=int, default=500, help="log loss every n steps")

    # data args
    # TODO: load a data args yaml file
    parser.add_argument(
        "--data_path",
        default="/export/home/LLaVA/playground/data/llava_v1_5_mix665k_ocr_tagged_vqa_placeholder.json",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data_sampler_group_by_length", default=False, action="store_true")

    # Legacy Llava data args
    parser.add_argument("--is_multimodal", type=bool, default=True)
    parser.add_argument("--mm_use_im_start_end", default=False, action="store_true")
    parser.add_argument("--conv_template_name", type=str, default=None)

    # Any resolution
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument(
        "--anyres_patch_sampling",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--anyres_grids",
        type=parse_tuple_list,
        default="(1,2),(2,1),(2,2),(3,1),(1,3)",
        help="List of tuples in the format (1,2),(3,4),...",
    )

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Local rank for distributed training")

    # fsdp args
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training. Not supported for some models, e.g. OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        default="full",
        type=str,
        choices=["full", "hybrid", "shard_grad_op", "hybrid_shard_grad_op", "no_shard"],
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument(
        "--dryrun",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--use_flash_attention_2",
        default=False,
        action="store_true",
        help="Use Flash Attention 2.0 for language model.",
    )
    parser.add_argument(
        "--unfreeze_vision_encoder", default=False, action="store_true", help="Unfreeze vision encoder during training."
    )
    parser.add_argument(
        "--vision_encoder_precision",
        default="fp32",
        choices=["bf16", "fp32"],
        help="Precision of the vision encoder during training.",
    )
    parser.add_argument(
        "--cpu_offload_gradients",
        default=False,
        action="store_true",
        help="This specifies whether to offload parameters to CPU when not involved in computation. If True, then this offloads gradients to CPU as well, meaning that the optimizer step runs on CPU.",
    )
    return parser.parse_args()


def parse_tuple_list(val: str) -> List[Tuple[int, int]]:
    return [tuple(map(int, x.strip("()").split(","))) for x in val.split(",") if x]


