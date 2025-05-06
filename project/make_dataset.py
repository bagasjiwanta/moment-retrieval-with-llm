# turns a qvh and charades style dataset into llava
"""Structure
datasets/<charades-sta or qvhighlights>
    annotations/
        raw/

        processed/
    videos/
        raw/
            train/ val/ test/
        processed/
            train/ val/ test/
"""
# QVH style:
"""
{'qid': 10016,
 'query': 'Man in baseball cap eats before doing his interview.',
 'duration': 150,
 'vid': 'j7rJstUseKg_210.0_360.0',
 'relevant_clip_ids': [48, 49, 50, 51, 52, 53, 54, 55, 56],
 'saliency_scores': [[2, 3, 3],
  [4, 3, 2],
  [2, 3, 1],
  [2, 3, 0],
  [2, 3, 3],
  [2, 3, 2],
  [2, 3, 1],
  [2, 3, 0],
  [1, 3, 3]],
 'relevant_windows': [[96, 114]]}
"""

# Charades style:
# AO8RW 0.0 6.9##a person is putting a book on a shelf.

# llava_style
"""
{
 "id": "000000033471",
 "image": "coco/train2017/000000033471.jpg",
 "conversations": [
 {
 "from": "human",
 "value": "<image>\nWhat are the colors of the bus in the image?"
 },
 {
 "from": "gpt",
 "value": "The bus in the image is white and red."
 },
 {
 "from": "human",
 "value": "What feature can be seen on the back of the bus?"
 },
 {
 "from": "gpt",
 "value": "The back of the bus features an advertisement."
 },
 ...
 ]
}
"""
import os
import argparse
from os.path import join as pjoin
import json
import numpy as np
import torchvision
from typing import List
from PIL import ImageFile, Image
from dataclasses import dataclass
from tqdm import tqdm
import random
from decord import VideoReader
from decord import cpu, gpu
import multiprocessing
from functools import partial
import re


BASE_PROMPT_TEMPLATE = "{video}\n{prompt}"
PROMPT_STYLES = [
    {
        "instruction": "Instruction: You are given {num_frames} images sample from a video. "
        "Identify the images containing the activity: {activity}"
        "\nOutput exactly {num_frames} characters using 0 or 1 with "
        "each character as a frame indicator. Write 1 "
        "if the activity appears, and 0 if it does not. "
        "No quotes, no extra text, no line breaks.",
        "example": "Example: number of frames = {num_frames}, answer = {demo}.\n ",
    },
    {
        "instruction": "Number of frames: {num_frames}.\nQuery: {activity}\nQuestion: Which frames contains the activity?",
        "example": None,
    },
]


@dataclass
class QVHighlightsData:
    qid: int
    query: str
    duration: int
    vid: str
    relevant_clip_ids: List[int]
    saliency_scores: List[List[int]]
    relevant_windows: List[List[int]]


class Args:
    prompt_style: str
    dataset_dir: str
    dataset_name: str
    num_frames: int
    num_workers: int


def json_dumps(data, indent=2, max_inline_length=120):
    json_str = json.dumps(data, indent=indent)
    # regex to find lists with no nested braces/brackets
    pattern = re.compile(r"\[\s*([^\[\]\{\}]+?)\s*\]", re.DOTALL)

    def replacer(match):
        content = match.group(1)
        # remove whitespace and newlines inside the list
        inline = " ".join(content.split())
        if len(inline) <= max_inline_length:
            return f"[ {inline} ]"
        else:
            return match.group(0)

    return pattern.sub(replacer, json_str)


def process_one_video(video_name: str, vid_in_dir: str, vid_out_dir: str, num_frames: int):
    filename = os.path.join(vid_in_dir, f"{video_name}.mp4")
    with open(filename, "rb") as f_in:
        vr = VideoReader(f_in, ctx=cpu())

    step = len(vr) / num_frames
    frame_indices = np.linspace(round(step / 2), len(vr) - round(step / 2), num_frames, dtype=int)
    video: np.ndarray = vr.get_batch(frame_indices).asnumpy()
    image_out_filenames = []

    if not os.path.isfile(os.path.join(vid_out_dir, f"{video_name}_frame{len(video):03d}.jpg")):
        print(os.path.join(vid_out_dir, f"{video_name}_frame{len(video):03d}.jpg"))
        for i in range(len(video)):
            frame = Image.fromarray(video[i])
            out_filename = os.path.join(vid_out_dir, f"{video_name}_frame{(i+1):03d}.jpg")
            frame.save(out_filename)
            image_out_filenames.append(os.path.basename(out_filename))

    video_times = vr.get_frame_timestamp(frame_indices).mean(-1).astype(int)
    return video_name, {"video_times": video_times, "image_out_filenames": image_out_filenames}


def process_one_qvh(
    data: dict,
    num_frames: int,
    prompt_style: int,
    video_summaries: dict
):

    video_summary = video_summaries[data['qid']]
    video_times = video_summary['video_times']
    image_out_filenames = video_summary['image_out_filenames']

    answer_times = np.array(data["relevant_windows"])
    binary_mask: np.ndarray = (
        np.any((video_times[:, None] >= answer_times[:, 0]) & (video_times[:, None] <= answer_times[:, 1]), axis=1)
        .astype(int)
        .tolist()
    )

    # make the prompts
    video_tokens = "<image>" * num_frames
    prompt = PROMPT_STYLES[prompt_style]
    activity = data["query"]
    if prompt["example"] is not None:
        demo = "".join(random.choice("01") for _ in range(num_frames))
        example = prompt["example"].format(demo=demo, num_frames=num_frames)
    else:
        example = ""
    gpt = "".join([str(b) for b in binary_mask])
    human = prompt["instruction"].format(
        num_frames=num_frames,
        activity=activity,
    )
    human = example + human
    human = BASE_PROMPT_TEMPLATE.format(prompt=human, video=video_tokens)
    conversations = [{"from": "human", "value": human}, {"from": "gpt", "value": gpt}]
    return (
        {
            "id": data["qid"],
            "image": image_out_filenames,
            "conversations": conversations,
            "video_timestamps": video_times.tolist(),
            "relevant_windows": data["relevant_windows"],
        },
    )


def process_qvh(dirs, num_frames: int, prompt_style: int, num_workers: int, pretty: bool = False):
    vid_in_dir = dirs["vid_in_dir"]
    ann_input_dir = dirs["ann_input_dir"]
    vid_out_dir = dirs["vid_out_dir"]
    ann_out_dir = dirs["ann_out_dir"]
    splits = ["train", "test", "val"]

    input_paths = [os.path.join(ann_input_dir, f"highlight_{split}_release.jsonl") for split in splits]
    assert all([os.path.isfile(path) for path in input_paths]), "QVHighlights dataset files are not found"
    output_paths = [os.path.join(ann_out_dir, f"highlight_{split}_release.json") for split in splits]

    for i in range(len(splits)):
        ann_in = input_paths[i]
        with open(ann_in, "r") as f_in:
            raw = [json.loads(line) for line in f_in.readlines()]
        video_names = list(set([r["vid"] for r in raw]))
        partial_process_one_video = partial(
            process_one_video, vid_in_dir=vid_in_dir, vid_out_dir=vid_out_dir, num_frames=num_frames
        )
        video_summaries = {}
        with multiprocessing.Pool(num_workers) as pool:
            iterator = tqdm(
                pool.imap(partial_process_one_video, video_names),
                total=len(video_names),
                desc=f"Processing {len(video_names)} videos",
            )
            for vid, summary in iterator:
                video_summaries[vid] = summary

        processed = []
        worker_func = partial(
            process_one_qvh,
            num_frames=num_frames,
            prompt_style=prompt_style,
            video_summaries=video_summaries
        )

        with multiprocessing.Pool(num_workers) as pool:
            for result in tqdm(pool.imap(worker_func, raw), total=len(raw), desc=f"Processing {ann_in}"):
                processed.append(result)

        ann_out = output_paths[i]
        print(len(processed))
        with open(ann_out, "w") as f_out:
            f_out.write(json_dumps(processed, indent=2 if pretty else None))


def process_charades(dirs, num_frames):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", default=0, type=int, choices=[0, 1])
    parser.add_argument("--dataset_dir", required=True, default="datasets", type=str)
    parser.add_argument("--dataset_name", default="qvhighlights", choices=["qvhighlights", "charades-sta"])
    parser.add_argument("--num_frames", type=int, default=16, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pretty_json", default=False, action="store_true")

    args: Args = parser.parse_args()
    vid_in_dir = os.path.join(args.dataset_dir, args.dataset_name, "videos", "raw")
    ann_input_dir = os.path.join(args.dataset_dir, args.dataset_name, "annotations", "raw")
    vid_out_dir = os.path.join(args.dataset_dir, args.dataset_name, "videos", "processed")
    ann_out_dir = os.path.join(args.dataset_dir, args.dataset_name, "annotations", "processed")
    print(f"Processing {args.dataset_name}\n")
    print(f"Args:")
    for k, v in vars(args).items():
        print(f" - {k}: {v}")
    dirs = dict(vid_in_dir=vid_in_dir, ann_input_dir=ann_input_dir, vid_out_dir=vid_out_dir, ann_out_dir=ann_out_dir)
    for k, v in dirs.items():
        print(f" - {k}: {v}")
    print("\n")
    if args.dataset_name == "qvhighlights":
        process_qvh(
            dirs, args.num_frames, prompt_style=args.prompt_style, num_workers=args.num_workers, pretty=args.pretty_json
        )
    else:
        process_charades(dirs, args.num_frames)


if __name__ == "__main__":
    main()
