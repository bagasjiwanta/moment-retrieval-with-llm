"""Main training script"""

import argparse
from datetime import datetime
import os
from omegaconf import OmegaConf
import torch
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


def find_all_linear_names(model):
    """
    Returns a list of all nn.Linear module names in the model
    excluding any names containing specified multimodal keywords and 'lm_head'.
    This list can be passed directly as target_modules for LoRA.
    """
    cls = torch.nn.Linear
    multimodal_keywords = ["vision_encoder"]
    lora_module_names = []

    for name, module in model.named_modules():
        # skip multimodal or tokenizer-specific submodules if desired
        if any(keyword in name for keyword in multimodal_keywords):
            continue
        # collect full module path for nn.Linear layers
        if isinstance(module, cls):
            # skip lm_head for 16-bit compatibility
            if name.endswith("lm_head"):
                continue
            lora_module_names.append(name)

    return lora_module_names


def parse_tuple_list(input_string):
    try:
        tuples = input_string.strip().strip("()").split("),(")
        # Convert each item in the list to a tuple
        tuple_list = [tuple(map(int, item.split(","))) for item in tuples]
        return tuple_list
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple list format: {input_string}. Error: {e}")


def main():
    args = parse_args()
    arg_dict = vars(args)
    print("================== Parsed Arguments ===================")
    for k, v in sorted(list(arg_dict.items())):
        print(f"{k}: {v}")
    print("=======================================================")

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.fsdp:
        assert torch.__version__ > "2.0.1", "FSDP requires torch > 2.0.1"

    # Set up distributed training
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    if args.rank == 0:
        print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize model
    if args.model_family in ["xgenmm_v1"]:
        additional_kwargs = {
            "image_aspect_ratio": args.image_aspect_ratio,
            "num_vision_tokens": args.num_vision_tokens,
            "anyres_patch_sampling": args.anyres_patch_sampling,
        }
    else:
        additional_kwargs = {}

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        lang_model_path=args.lm_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.lm_path,
        model_family=args.model_family,
        pretrained_vision_tokenizer=args.pretrained_vision_tokenizer,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        verbose=(args.rank == 0),
        anyres_grids=args.anyres_grids,
        use_flash_attention_2=args.use_flash_attention_2,
        **additional_kwargs,
    )
    random_seed(args.seed, args.rank)

    # Initialize wandb logging
    now = datetime.now().strftime("%Y%m%d%H%M")[:-1]
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.run_name}-{now}",
            config=vars(args),
        )

    # Load model checkpoint (on CPU)
    if args.fsdp:
        args.fsdp_checkpoint_config = get_fsdp_checkpoint_config(args)

    # if args do not specify a checkpoint to resume from, resume from most recent checkpoint
    resume_from_step = 0
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = find_most_recent_checkpoint(args)

    if args.resume_from_checkpoint is not None:
        resume_from_epoch, resume_from_step, checkpoint = load_checkpoint(args, model)
        print(f"Resume training from epoch {resume_from_epoch}, step {resume_from_step}...")
    else:
        resume_from_epoch = 0
        resume_from_step = 0

    # Load pretrained weights.
    if args.resume_from_checkpoint is None and not args.dryrun:
        if args.pretrained_vision_tokenizer is None:
            assert os.path.exists(args.pretrained), "Must fine-tune from a pretrained weight."
        if args.pretrained is not None:
            _, _, checkpoint = load_checkpoint(args, model, pretrained=True)
            print("Finished loading checkpoint...")

    print("Wrapping model in LoRA")
    if args.lora:
        linear_names = find_all_linear_names(model)
        print("================== Trainable Parameters ===================")
        print("\n".join(linear_names))
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )
        print("===========================================================")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        model.init_gradient_checkpointing()

    # Initialize FSDP / DDP, and ensure the model is on GPU
    if args.fsdp:
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=model.get_fsdp_lambda_fn())
        wrapper_kwargs = get_fsdp_config(args, device_id)
        distributed_model = FSDP(model, auto_wrap_policy=auto_wrap_policy, **wrapper_kwargs)
        print("Finished FSDP wrapping...")
    else:
        model = model.to(device_id).to(torch.bfloat16)
        distributed_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # Initialize optimizer
    params_with_wd, params_without_wd = model.group_params_by_weight_decay()
    optimizer = torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        optim_state_dict = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            # FSDP.set_state_dict_type(
            #     distributed_model,
            #     **args.fsdp_checkpoint_config,
            # )
            optim_state_dict = FSDP.optim_state_dict_to_load(
                model=distributed_model, optim=optimizer, optim_state_dict=optim_state_dict
            )
        optimizer.load_state_dict(optim_state_dict)

    # Initialize datasets
    if args.data_path.split(".")[-1] == "yaml":
        # Loading a mixture of datasets with sampling ratios.
        data_config = OmegaConf.load(args.data_path)
        if args.rank == 0:
            print("================== Data mixture config ===================")
            print(data_config)
            print("==========================================================")
        args.data_path = dict(data_config.data_path)
    train_dataset, total_num_samples = make_supervised_data_module(tokenizer, image_processor, args, train=True)
    val_dataset, val_samples = make_supervised_data_module(tokenizer, image_processor, args, train=False)
    
    # Update anyres grid.
    args.anyres_grids = train_dataset.dataloader.dataset.anyres_grids
    model.anyres_grids = args.anyres_grids  # this does not seem to work idk, the grids are inserted above

    # TODO: Summarize training data stats (dataset, portion, etc.)
    total_training_steps = (
        total_num_samples // (args.batch_size * args.gradient_accumulation_steps * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Initialize the loss fn
    loss_fn = get_loss_fn(args.loss)

    # check wrapping
    if args.rank == 0:
        print(distributed_model)

    # Start training!
    print(f"Start running training on rank {args.rank}.")
    for epoch in range(resume_from_epoch, args.num_epochs):
        train_dataset.set_epoch(epoch)
        finetune_one_epoch(
            args=args,
            resume_from_step=resume_from_step,
            model=distributed_model,
            epoch=epoch,
            dataset=train_dataset,
            compute_loss_fn=loss_fn,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device_id,
            wandb=wandb,
        )

        save_checkpoint(distributed_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()
