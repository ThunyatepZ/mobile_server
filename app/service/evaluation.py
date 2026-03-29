"""
ROUGE Evaluation Service
ใช้สำหรับประเมินคุณภาพคำตอบของ RAG เทียบกับ Reference Answer
"""

from typing import List, Optional
from rouge_score import rouge_scorer


def evaluate_single(reference: str, hypothesis: str) -> dict:
    """
    ประเมินคำตอบ 1 คู่ด้วย ROUGE
    
    Args:
        reference: คำตอบที่ถูกต้อง (ground truth)
        hypothesis: คำตอบที่ RAG สร้างขึ้น
    
    Returns:
        dict ที่มี rouge1, rouge2, rougeL scores
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,  # ภาษาไทยไม่ต้องใช้ stemmer
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1": {
            "precision": round(scores["rouge1"].precision, 4),
            "recall": round(scores["rouge1"].recall, 4),
            "fmeasure": round(scores["rouge1"].fmeasure, 4),
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision, 4),
            "recall": round(scores["rouge2"].recall, 4),
            "fmeasure": round(scores["rouge2"].fmeasure, 4),
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision, 4),
            "recall": round(scores["rougeL"].recall, 4),
            "fmeasure": round(scores["rougeL"].fmeasure, 4),
        },
    }


def evaluate_batch(pairs: List[dict]) -> dict:
    """
    ประเมินหลายคู่พร้อมกัน แล้วหาค่าเฉลี่ย
    
    Args:
        pairs: list of {"reference": str, "hypothesis": str}
    
    Returns:
        dict ที่มี individual_scores, average_scores, num_pairs
    """
    if not pairs:
        return {"error": "ไม่มีข้อมูลให้ประเมิน", "individual_scores": [], "average_scores": {}}

    individual_scores = []
    totals = {
        "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
        "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
        "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0},
    }

    for pair in pairs:
        ref = pair.get("reference", "")
        hyp = pair.get("hypothesis", "")
        score = evaluate_single(ref, hyp)
        individual_scores.append(score)

        for metric in totals:
            for key in totals[metric]:
                totals[metric][key] += score[metric][key]

    n = len(pairs)
    average_scores = {}
    for metric in totals:
        average_scores[metric] = {
            key: round(val / n, 4) for key, val in totals[metric].items()
        }

    return {
        "num_pairs": n,
        "individual_scores": individual_scores,
        "average_scores": average_scores,
    }
