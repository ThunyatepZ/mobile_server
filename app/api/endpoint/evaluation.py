"""
Evaluation API Endpoint
ใช้สำหรับประเมินคุณภาพ RAG ด้วย ROUGE
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

from app.api.endpoint.auth import decoder_token, oauth2_scheme
from app.service.chaobot import ask_chatbot
from app.service.evaluation import evaluate_single, evaluate_batch

router = APIRouter()


# ────────────────────────────────────────────────────────────
# Request / Response Models
# ────────────────────────────────────────────────────────────
class EvalSingleRequest(BaseModel):
    """ประเมิน 1 คู่: ส่ง question + reference_answer → ระบบสร้างคำตอบจาก RAG แล้วเทียบ"""
    question: str
    reference_answer: str


class EvalDirectRequest(BaseModel):
    """เทียบ ROUGE ตรงๆ ระหว่าง reference กับ hypothesis (ไม่ต้องผ่าน RAG)"""
    reference: str
    hypothesis: str


class EvalBatchRequest(BaseModel):
    """ประเมินหลายคู่พร้อมกัน"""
    pairs: List[EvalDirectRequest]


class EvalWithRAGBatchRequest(BaseModel):
    """ประเมินหลายคำถามผ่าน RAG พร้อมกัน"""
    questions: List[EvalSingleRequest]


# ────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────
@router.post("/rouge")
def evaluate_rouge_direct(request: EvalDirectRequest):
    """
    เทียบ ROUGE ตรงๆ ระหว่าง reference กับ hypothesis
    ไม่ต้องผ่าน RAG — ใช้สำหรับทดสอบตรงๆ
    """
    result = evaluate_single(request.reference, request.hypothesis)
    return {"success": True, "scores": result}


@router.post("/rouge-batch")
def evaluate_rouge_batch(request: EvalBatchRequest):
    """
    ประเมินหลายคู่พร้อมกัน (reference vs hypothesis)
    ส่งกลับทั้ง individual scores และค่าเฉลี่ย
    """
    pairs = [{"reference": p.reference, "hypothesis": p.hypothesis} for p in request.pairs]
    result = evaluate_batch(pairs)
    return {"success": True, **result}


@router.post("/rag-rouge")
def evaluate_rag_with_rouge(
    request: EvalSingleRequest,
    token: str = Depends(oauth2_scheme),
):
    """
    ส่ง question + reference_answer เข้ามา
    → ระบบสร้างคำตอบจาก RAG
    → เทียบกับ reference_answer ด้วย ROUGE
    → ส่ง scores กลับ
    """
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    session_id = token_data.email

    # สร้างคำตอบจาก RAG
    rag_answer = ask_chatbot(session_id=session_id, question=request.question)

    # ประเมินด้วย ROUGE
    scores = evaluate_single(request.reference_answer, rag_answer)

    return {
        "success": True,
        "question": request.question,
        "rag_answer": rag_answer,
        "reference_answer": request.reference_answer,
        "scores": scores,
    }


@router.post("/rag-rouge-batch")
def evaluate_rag_with_rouge_batch(
    request: EvalWithRAGBatchRequest,
    token: str = Depends(oauth2_scheme),
):
    """
    ส่งหลายคำถาม + reference_answer เข้ามาพร้อมกัน
    → ระบบสร้างคำตอบจาก RAG ทุกคำถาม
    → เทียบแต่ละคู่ด้วย ROUGE
    → ส่ง individual scores + ค่าเฉลี่ยกลับ
    """
    try:
        token_data = decoder_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    session_id = token_data.email

    results = []
    pairs_for_batch = []

    for q in request.questions:
        rag_answer = ask_chatbot(session_id=session_id, question=q.question)
        results.append({
            "question": q.question,
            "rag_answer": rag_answer,
            "reference_answer": q.reference_answer,
        })
        pairs_for_batch.append({
            "reference": q.reference_answer,
            "hypothesis": rag_answer,
        })

    batch_scores = evaluate_batch(pairs_for_batch)

    # รวม individual scores เข้ากับ results
    for i, result in enumerate(results):
        result["scores"] = batch_scores["individual_scores"][i]

    return {
        "success": True,
        "num_questions": len(results),
        "results": results,
        "average_scores": batch_scores["average_scores"],
    }
