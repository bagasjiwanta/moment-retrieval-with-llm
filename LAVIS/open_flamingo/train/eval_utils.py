import torch
from eval_qvhighlights import eval_submission
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from typing import TypedDict, Iterator, Tuple
from train_utils import DataInfo, get_autocast
from args import Args
from losses import unwrap_model
from sft_data_utils import DataCollatorOutput


class EvalDataDict(TypedDict):
    input_ids: torch.Tensor
    duration: int
    qid: int
    vid: str


def preds_to_spans(data_dict: List[EvalDataDict]) -> List[List[int]]:
    """
    preds: list of strings, each string is a series of '0'/'1'
    durations: list of ints, total duration (in seconds) for each video
    Returns:
        list of list of [start, end] for each pred
    """
    preds = [data_dict[i]["preds"] for i in range(len(data_dict))]
    durations = [data_dict[i]["duration"] for i in range(len(data_dict))]
    for i, (pred, duration) in enumerate(zip(preds, durations)):
        N = len(pred)
        time_step = duration / N
        spans = []
        in_span = False
        for i, val in enumerate(pred):
            if val == "1" and not in_span:
                in_span = True
                start = int(i * time_step)
            elif val == "0" and in_span:
                end = int(i * time_step)
                spans.append([start, end])
                in_span = False
        # Handle case where span goes to end
        if in_span:
            end = int(duration)  # last span closes at video end
            spans.append([start, end])
        data_dict[i]["pred_relevant_windows"] = spans

    return data_dict


def preds_to_spans_np(preds, durations):
    """unused"""
    all_spans = []
    for pred, dur in zip(preds, durations):
        arr = np.array(list(pred), dtype=int)
        n = len(arr)
        step = dur / n
        padded = np.pad(arr, (1, 1), "constant")
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        spans = []
        for s, e in zip(starts, ends):
            start_sec = int(s * step)
            end_sec = int(e * step)
            spans.append([start_sec, end_sec])
        all_spans.append(spans)
    return all_spans


def validate_one_epoch(args: Args, model, epoch, dataset: DataInfo, tokenizer, device_id, wandb):
    num_batches_per_epoch = len(dataset.dataloader)
    model.eval()
    iterator: Iterator[Tuple[int, DataCollatorOutput]] = tqdm(
        enumerate(dataset.dataloader),
        disable=args.rank != 0,
        total=len(dataset.dataloader),
        initial=args.num_epochs * num_batches_per_epoch,
    )

    preds = []
    for step_num, samples in iterator:
        images = samples["images"]
        if not isinstance(images, list):
            images = images.to(device_id, non_blocking=True)

        input_ids = samples["input_ids"].to(device_id, non_blocking=True)
        attention_mask = samples["attention_mask"].to(device_id, non_blocking=True)
        generated_ids = model.generate(
            vision_x=images,
            lang_x=input_ids,
            image_size=samples["image_size"],
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=256,
            top_p=0.9,
            num_beams=1,
        )
        # B, T
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.split("<|end|>")[0]
        metadata = samples["metadata"]
        for i in range(args.batch_size):
            metadata[i]["preds"] = generated_text
        preds.append(metadata)
        break

    spans = preds_to_spans(preds)
    print({})


class EvalDataDict(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor
    duration: int
    qid: int
    vid: str
