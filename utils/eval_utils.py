"""
Evaluation utilities for visual web understanding tasks.

This module provides evaluation functions for various tasks in the VisualWebBench
benchmark including web captioning, OCR tasks, element/action grounding, and
visual question answering. Each function computes task-specific metrics such as
ROUGE scores, accuracy, precision, or F1 scores.
"""
import re

import numpy as np
from rouge import Rouge

import torch
from torchvision.ops import box_iou


def eval_web_caption(preds, golds, **kwargs):
    """
    Evaluate web caption generation using ROUGE metrics.

    This function computes ROUGE-1, ROUGE-2, and ROUGE-L scores for web caption
    generation tasks. It handles empty predictions by replacing them with spaces.

    Args:
        preds: List of predicted captions.
        golds: List of ground truth captions.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing ROUGE scores scaled to 0-100 range.
            Keys: 'rouge_1', 'rouge_2', 'rouge_l'
    """
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores["rouge-1"]["f"] * 100,
        rouge_2=scores["rouge-2"]["f"] * 100,
        rouge_l=scores["rouge-l"]["f"] * 100,
    )


def eval_heading_ocr(preds, golds, **kwargs):
    """
    Evaluate heading OCR using ROUGE metrics.

    This function computes ROUGE-1, ROUGE-2, and ROUGE-L scores for heading
    text recognition tasks. It handles empty predictions by replacing them with spaces.

    Args:
        preds: List of predicted heading texts.
        golds: List of ground truth heading texts.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing ROUGE scores scaled to 0-100 range.
            Keys: 'rouge_1', 'rouge_2', 'rouge_l'
    """
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores["rouge-1"]["f"] * 100,
        rouge_2=scores["rouge-2"]["f"] * 100,
        rouge_l=scores["rouge-l"]["f"] * 100,
    )


def eval_element_ocr(preds, golds, **kwargs):
    """
    Evaluate element OCR using ROUGE metrics.

    This function computes ROUGE-1, ROUGE-2, and ROUGE-L scores for element
    text recognition tasks. It handles empty or single-character predictions
    by replacing them with spaces.

    Args:
        preds: List of predicted element texts.
        golds: List of ground truth element texts.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing ROUGE scores scaled to 0-100 range.
            Keys: 'rouge_1', 'rouge_2', 'rouge_l'
    """
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores["rouge-1"]["f"] * 100,
        rouge_2=scores["rouge-2"]["f"] * 100,
        rouge_l=scores["rouge-l"]["f"] * 100,
    )


def eval_action_prediction(preds, golds, **kwargs):
    """
    Evaluate action prediction as a multiple choice task.

    This function parses multiple choice responses and computes accuracy
    for action prediction tasks. It converts letter choices to indices
    and compares against ground truth indices.

    Args:
        preds: List of predicted action choices (letters A-H).
        golds: List of ground truth action indices (0-7).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing accuracy scaled to 0-100 range.
            Keys: 'accuracy'
    """
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(
            pred, [chr(ord("A") + i) for i in range(8)]
        )
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_element_ground(preds, golds, **kwargs):
    """
    Evaluate element grounding as a multiple choice task.

    This function parses multiple choice responses and computes accuracy
    for element grounding tasks. It converts letter choices to indices
    and compares against ground truth indices.

    Args:
        preds: List of predicted element choices (letters A-H).
        golds: List of ground truth element indices (0-7).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing accuracy scaled to 0-100 range.
            Keys: 'accuracy'
    """
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(
            pred, [chr(ord("A") + i) for i in range(8)]
        )
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_action_ground(preds, golds, **kwargs):
    """
    Evaluate action grounding as a multiple choice task.

    This function parses multiple choice responses and computes accuracy
    for action grounding tasks. It converts letter choices to indices
    and compares against ground truth indices.

    Args:
        preds: List of predicted action choices (letters A-H).
        golds: List of ground truth action indices (0-7).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing accuracy scaled to 0-100 range.
            Keys: 'accuracy'
    """
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(
            pred, [chr(ord("A") + i) for i in range(8)]
        )
        try:
            if ord("A") <= ord(cur_pred) <= ord("Z"):
                cur_pred = ord(cur_pred) - ord("A")
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(accuracy=sum(results) / len(results) * 100)


def eval_element_bbox_ground(preds, golds, **kwargs):
    """
    Evaluate element bounding box grounding using IoU precision.

    This function computes the precision of bounding box predictions by
    calculating IoU (Intersection over Union) with ground truth boxes.
    A prediction is considered correct if IoU >= 0.5.

    Args:
        preds: List of predicted bounding boxes in format (x1, y1, x2, y2).
        golds: List of ground truth bounding boxes in format (x1, y1, x2, y2).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing precision scaled to 0-100 range.
            Keys: 'precision'
    """
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0.0, 0.0, 0.0, 0.0)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(precision=correct / total_cnt * 100)


def eval_action_bbox_ground(preds, golds, **kwargs):
    """
    Evaluate action bounding box grounding using IoU precision.

    This function computes the precision of bounding box predictions by
    calculating IoU (Intersection over Union) with ground truth boxes.
    A prediction is considered correct if IoU >= 0.5.

    Args:
        preds: List of predicted bounding boxes in format (x1, y1, x2, y2).
        golds: List of ground truth bounding boxes in format (x1, y1, x2, y2).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing precision scaled to 0-100 range.
            Keys: 'precision'
    """
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0.0, 0.0, 0.0, 0.0)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(precision=correct / total_cnt * 100)


def eval_webqa(preds, golds, **kwargs):
    """
    Evaluate web question answering using maximum F1 score.

    This function computes F1 scores for web QA tasks where each question
    may have multiple acceptable answers. It takes the maximum F1 score
    across all possible answers for each question.

    Args:
        preds: List of predicted answers.
        golds: List of lists of ground truth answers (multiple answers per question).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        dict: Dictionary containing F1 score scaled to 0-100 range.
            Keys: 'f1'
    """
    f1_scores = []
    rouge = Rouge(metrics=["rouge-1"])
    for pred, gold_list in zip(preds, golds):
        try:
            if not pred:
                pred = " "
            cur_f1 = max(
                [
                    rouge.get_scores([pred], [gold], avg=True)["rouge-1"]["f"]
                    for gold in gold_list
                ]
            )
            f1_scores.append(cur_f1)
        except:
            pass

    return dict(f1=sum(f1_scores) / len(f1_scores) * 100)


def eval_element_point_ground(preds, golds):
    """
    Evaluate element point grounding using containment accuracy.

    This function checks if predicted points fall within the ground truth
    bounding boxes. A prediction is correct if the point is contained
    within the target bounding box.

    Args:
        preds: List of predicted points in format (x, y).
        golds: List of ground truth bounding boxes in format (left, top, right, bottom).

    Returns:
        dict: Dictionary containing accuracy scaled to 0-100 range.
            Keys: 'accuracy'
    """
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left <= x <= right and top <= y <= bottom)
    return dict(accuracy=sum(acc_lst) / len(acc_lst) * 100)


def eval_action_point_ground(preds, golds):
    """
    Evaluate action point grounding using containment accuracy.

    This function checks if predicted points fall within the ground truth
    bounding boxes. A prediction is correct if the point is contained
    within the target bounding box.

    Args:
        preds: List of predicted points in format (x, y).
        golds: List of ground truth bounding boxes in format (left, top, right, bottom).

    Returns:
        dict: Dictionary containing accuracy scaled to 0-100 range.
            Keys: 'accuracy'
    """
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left <= x <= right and top <= y <= bottom)
    return dict(accuracy=sum(acc_lst) / len(acc_lst) * 100)


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response: str, all_choices):
    """
    Parse multiple choice responses to extract the predicted answer.

    This function handles various formats of multiple choice responses,
    including single letters, bracketed choices, and responses with
    punctuation. It uses pattern matching and position analysis to
    extract the most likely intended answer.

    Args:
        response: The model's response string.
        all_choices: List of valid choice letters (e.g., ['A', 'B', 'C', 'D']).

    Returns:
        str: The predicted choice letter, or 'z' if no valid choice is found.
    """
    # Handle simple single letter responses
    if len(response) == 1:
        return response.upper()
    elif not response:
        return "a"
    elif re.match(r"[A-Z]\.", response):
        return response[0]

    # Clean punctuation from response
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " "  # add space to avoid partial match

    ans_with_brack = False
    candidates = []

    # Look for bracketed choices like (A), (B), etc.
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # If no bracketed choices found, look for spaced choices like " A ", " B ", etc.
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Handle case where no valid choices are found
    if len(candidates) == 0:
        pred_index = "z"
    # Handle case where multiple choices are found - take the last one
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                index = response.rfind(f"({can})")
                start_indexes.append(index)  # -1 will be ignored anyway
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # Get the last occurrence
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # Single candidate found
        pred_index = candidates[0]

    return pred_index
