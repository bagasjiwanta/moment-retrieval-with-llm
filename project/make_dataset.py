# turns a qvh or charades style dataset into llava
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


BASE_PROMPT_TEMPLATE = """<video><example><instruction>"""
PROMPT_STYLES = [
    {
        "example": "Example: number of frames = {num_frames}, answer = {demo}",
        "instruction": "You are given {num_frames} images sample from a video. "
        "Identify the images containing the activity: {activity}"
        "\nOutput exactly {num_frames} characters using 0 or 1 with "
        "each character as a frame indicator. Write 1 "
        "if the activity appears, and 0 if it does not. "
        "No quotes, no extra text, no line breaks.",
    },
    {"example": "", "instruction": "Frame: {num_frame}. Query: When does {activity} happens in the video?"},
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
    output_dir: str
    input_dir: str
    dataset_name: str
    num_frames: int


def process_one_qvh(data: dict, vid_in_dir: str, vid_out_dir: str, num_frames: int):
    filename = os.path.join(vid_in_dir, f"{data['vid']}.mp4")
    vframes, _, info = torchvision.io.read_video(filename=filename, pts_unit="sec", output_format="TCHW")

    fps = float(info["video_fps"])
    frame_indices = np.linspace(int(num_frames / 2), len(vframes) - int(num_frames / 2), num_frames, dtype=int)
    video = vframes[frame_indices]

    for i in range(len(video)):
        frame = torchvision.transforms.functional.to_pil_image(video[i])
        out_filename = os.path.join(vid_out_dir, f"{data['vid']}_frame{(i+1):03d}.jpg")
        frame.save(out_filename)

    video_times = (frame_indices / fps).tolist()
    video_times = map(round, video_times)  # convert to integers
    answer_times = np.array(data["relevant_windows"])
    binary_mask = np.any(
        (video_times[:, None] >= answer_times[:, 0]) & (video_times[:, None] <= answer_times[:, 1]), axis=1
    ).astype(int)
    return video_times, binary_mask


def process_qvh(dirs, num_frames):
    vid_in_dir = dirs["vid_in_dir"]
    ann_input_dir = dirs["ann_input_dir"]
    vid_out_dir = dirs["vid_out_dir"]

    paths = [os.path.join(ann_input_dir, f"highlight_{split}_release.jsonl") for split in ["train", "test", "val"]]
    assert all([os.path.isfile(path) for path in paths]), "QVHighlights dataset files are not found"

    with open(paths[0], "r") as f_in:
        qvh_train = [json.loads(line) for line in f_in.readlines()]
    with open(paths[1], "r") as f_in:
        qvh_test = [json.loads(line) for line in f_in.readlines()]
    with open(paths[2], "r") as f_in:
        qvh_val = [json.loads(line) for line in f_in.readlines()]

    train = []
    for s in qvh_train:
        process_one_qvh(s, vid_in_dir, vid_out_dir, num_frames=num_frames)


def process_charades(dirs, num_frames):
    pass 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", default=0, type=int, choices=[0, 1])
    parser.add_argument("--output_dir", required=True, default="out", type=str)
    parser.add_argument("--input_dir", required=True, default="", type=str)
    parser.add_argument("--dataset_name", required=True, default="qvhighlights", choices=["qvhighlights", "charades"])
    parser.add_argument("--num_frames", type=int, default=16, required=True)

    args: Args = parser.parse_args()
    vid_in_dir = os.path.join(args.input_dir, "videos")
    ann_input_dir = os.path.join(args.input_dir, "annotations")
    vid_out_dir = os.path.join(args.input_dir, "videos")
    ann_input_dir = os.path.join(args.input_dir, "annotations")

    dirs = dict(vid_in_dir=vid_in_dir, ann_input_dir=ann_input_dir, vid_out_dir=vid_out_dir)
    if args.dataset_name == "qvhighlights":
        process_qvh(dirs, args.num_frames)
    else:
        process_charades(dirs, args.num_frames)
