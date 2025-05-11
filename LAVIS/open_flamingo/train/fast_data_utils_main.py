import argparse
from dataclasses import dataclass
from tqdm import trange
from open_flamingo.src.factory import create_model_and_transforms
import time
from open_flamingo.train.sft_data_utils import LazySupervisedDataset
from open_flamingo.train.train_utils import random_seed


@dataclass
class QvhDataArgs:
    data_path = ("a",)
    image_aspect_ratio = "anyres"
    conv_template_name = "phi_3"
    anyres_grids = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]
    data_config = {
        "qvhighlights": {
            "train": {
                "annotations": {
                    "../datasets/qvhighlights/annotations/processed/highlight_train_release.json": 1000,
                },
                "videos": "../datasets/qvhighlights/videos/processed",
            },
            "val": {
                "annotations": {
                    "../datasets/qvhighlights/annotations/processed/highlight_val_release.json": 207,
                },
                "videos": "../datasets/qvhighlights/videos/processed",
            },
        }
    }


def test_retrieval_speed(dataset_name="qvhighlights"):
    model, image_processor, text_tokenizer = create_model_and_transforms(
        vision_encoder_path="google/siglip-so400m-patch14-384",
        lang_model_path="microsoft/Phi-3-mini-4k-instruct",
        anyres_grids=[(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
        model_family="xgenmm_v1",
        pretrained_vision_tokenizer=None,
        use_local_files=False,
        verbose=True,
        use_flash_attention_2=True,
        image_aspect_ratio="anyres",
        num_vision_tokens=128,
        anyres_patch_sampling=True,
        gradient_checkpointing=True,
    )
    random_seed(42)
    del model

    data_args = QvhDataArgs() if dataset_name == "qvhighlights" else "charades-sta"
    lazy_train_dataset = LazySupervisedDataset(
        tokenizer=text_tokenizer,
        image_processor=image_processor,
        split="train",
        data_args=data_args,
        data_config=data_args.data_config,
    )

    lazy_val_dataset = LazySupervisedDataset(
        tokenizer=text_tokenizer,
        image_processor=image_processor,
        split="val",
        data_args=data_args,
        data_config=data_args.data_config,
    )
    print(f"Processing {len(lazy_train_dataset) + len(lazy_val_dataset)}")

    total_time_for_train = 0.0
    len_train_split = len(lazy_train_dataset)
    for i in trange(len(lazy_train_dataset), desc="processing train split"):
        start = time.time()
        sample = lazy_train_dataset[i]
        end = time.time()
        total_time_for_train += end - start

    print(f"{total_time_for_train=:2f} seconds, {len_train_split=}")

    total_time_for_val = 0.0
    len_val_split = len(lazy_val_dataset)
    for i in trange(len_val_split, desc="processing val split"):
        start = time.time()
        sample = lazy_val_dataset[i]
        end = time.time()
        total_time_for_val += end - start

    print(f"{total_time_for_val=:2f} seconds, {len_val_split=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", required=True, default="qvhighlights", choices=["qvhighlights", "charades-sta"]
    )
    args = parser.parse_args()
    test_retrieval_speed(dataset_name=args.dataset_name)
