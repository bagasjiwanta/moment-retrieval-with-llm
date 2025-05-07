import random
import copy
import json
from eval_qvhighlights import eval_submission
from eval_utils import preds_to_spans, preds_to_spans_np
import time


def mock_moment_retrieval_answer(sample, jitter=15, overlap_prob=0.75, false_positive_prob=0.35):
    """
    sample: dict as described in your example.
    jitter: maximum seconds to jitter start/end of relevant windows.
    overlap_prob: probability that the predicted window overlaps with ground-truth.
    false_positive_prob: probability of adding an extra window (false positive).
    Returns: list of [st, ed] predicted windows.
    """
    gt_windows = sample["relevant_windows"]
    duration = sample["duration"]

    preds = []
    for st, ed in gt_windows:
        # With high prob, generate overlapping window
        if random.random() < overlap_prob:
            st_jit = max(0, st + random.randint(-jitter, jitter))
            ed_jit = min(duration, ed + random.randint(-jitter, jitter))
            if ed_jit > st_jit:
                preds.append([st_jit, ed_jit, overlap_prob])
        # With low prob, totally miss this window

    # With some prob, add a false positive (spurious prediction)
    if random.random() < false_positive_prob:
        length = random.randint(5, 20)
        false_st = random.randint(0, duration - length)
        false_ed = false_st + length
        preds.append([false_st, false_ed, false_positive_prob])
    
    if len(preds) == 0:
        preds.append([50, 100, 0.5])

    return {
        "qid": sample["qid"],
        "query": sample["query"],
        "duration": sample["duration"],
        "vid": sample["vid"],
        "pred_relevant_windows": preds,
    }


def random_pred(length=100, p_on=0.2):
    """Create a random prediction string of given length, with probability p_on of '1'."""
    return ''.join(['1' if random.random() < p_on else '0' for _ in range(length)])


def batch_random_preds(n_preds=100, length=100, p_on=0.2, duration=150):
    """Create a batch of random prediction strings and durations."""
    preds = [random_pred(length, p_on) for _ in range(n_preds)]
    durations = [duration for _ in range(n_preds)]  # Fixed duration, can randomize if needed
    return preds, durations


def test_methods_for_val_epoch(n_preds=10000, length=100, p_on=0.2):
    preds, durations = batch_random_preds(n_preds=n_preds, length=length, p_on=p_on)
    # Sequential method
    t0 = time.time()
    out_seq = preds_to_spans(preds, durations)
    t1 = time.time()
    print(f"Sequential time: {t1 - t0:.4f}s")
    
    # Numpy method
    t0 = time.time()
    out_np = preds_to_spans_np(preds, durations)
    t1 = time.time()
    print(f"Numpy time: {t1 - t0:.4f}s")
    
    # Correctness check
    eq = out_seq == out_np
    print(f"Outputs match: {eq}")
    if not eq:
        for i, (a, b) in enumerate(zip(out_seq, out_np)):
            if a != b:
                print(f"Mismatch at index {i}: seq={a} np={b}")
                break

# # Example usage:
# sample = {
#     "qid": 5335,
#     "query": "A man in blue is cooking various slabs of beef on a flat cooktop.",
#     "duration": 150,
#     "vid": "8s9fLWEi4So_360.0_510.0",
#     "relevant_clip_ids": [30, 31, 32, 33, 34, 35, 36, 37],
#     "saliency_scores": [[3, 1, 2], [3, 4, 2], [3, 3, 2], [3, 2, 3], [3, 2, 3], [3, 2, 3], [3, 2, 3], [3, 2, 3]],
#     "relevant_windows": [[60, 76]],
# }


def test_eval_submission():
    random.seed(42)

    with open("datasets/qvhighlights/annotations/raw/highlight_val_release.jsonl", "r") as f_in:
        ground_truth = [json.loads(f) for f in f_in.readlines()]

    submission = []
    for i in range(len(ground_truth)):
        submission.append(mock_moment_retrieval_answer(ground_truth[i]))
    assert all([len(d["pred_relevant_windows"]) != 0 for d in submission])

    results = eval_submission(submission, ground_truth, verbose=False)
    print(dict(results['brief']))


def test_eval_loop():
    with open("datasets/qvhighlights/annotations/raw/highlight_val_release.jsonl", "r") as f_in:
        # accumulated ground truth
        ground_truth = [json.loads(f) for f in f_in.readlines()]
        
    preds, durations = batch_random_preds(n_preds=233, length=30, p_on=0.3)
    preds = preds_to_spans(preds, durations)
    print(preds[0])


if __name__ == "__main__":
    # test_eval_submission()
    # test_methods_for_val_epoch(length=30)
    test_eval_loop()