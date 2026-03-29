import io
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

PINECONE_INDEX_NAME = "learnify-docs"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_KEY"),
    model="typhoon-v2.5-30b-a3b-instruct",
    temperature=0.0,
    max_tokens=8192,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณคือ 'Learnify Bot' ผู้ช่วยส่วนตัวที่ฉลาดและเป็นมิตร หน้าที่ของคุณคือช่วยผู้ใช้เรียนรู้และทำความเข้าใจเนื้อหาต่างๆ\n\n"
        "ข้อควรปฏิบัติ:\n"
        "1. หากผู้ใช้อัปโหลดเอกสารมา (ดูจาก Context) ให้ตอบโดยอิงจากข้อมูลใน Context เป็นหลัก และห้ามเดาข้อมูลที่ไม่ได้อยู่ใน Context\n"
        "2. ถ้าคำถามเป็นข้อมูลเฉพาะเจาะจง เช่น ชื่ออาจารย์ รหัสวิชา หน่วยกิต ตารางเรียน วันเวลา ห้องเรียน รายชื่อ หรือรายละเอียดเชิงข้อเท็จจริงอื่นๆ แต่ Context ไม่มีหลักฐานชัดเจน ให้ตอบตรงๆว่า 'ไม่พบข้อมูลนี้ในเอกสารที่อัปโหลด' และชวนผู้ใช้อัปโหลดเอกสารที่เกี่ยวข้องเพิ่ม แทนการคาดเดา\n"
        "3. หากข้อมูลที่ถามไม่มีในเอกสาร หรือผู้ใช้ยังไม่ได้อัปโหลดเอกสาร คุณสามารถตอบโดยใช้ความรู้ทั่วไปที่คุณมีได้ตามความเหมาะสม แต่ต้องระบุให้ชัดว่าคำตอบส่วนนั้นเป็น 'ความรู้ทั่วไป' ไม่ใช่ข้อมูลจากเอกสาร\n"
        "3. ให้คำแนะนำด้วยน้ำเสียงที่เป็นมิตร กระตือรือร้น และส่งเสริมการเรียนรู้\n"
        "4. ให้ตอบแบบสรุปสั้น กระชับ เป็นค่าเริ่มต้น โดยปกติไม่เกิน 3-5 บรรทัด หรือ 3 bullet สั้นๆ\n"
        "5. ถ้าผู้ใช้ถามหาข้อมูลเฉพาะ เช่น ชื่ออาจารย์ รหัสวิชา จำนวนหน่วยกิต ให้ตอบเป็นคำตอบตรงๆก่อน ไม่ต้องเกริ่นยาว\n"
        "6. ให้ลงรายละเอียดเพิ่มเติมเฉพาะเมื่อผู้ใช้ขอ เช่น ขอแบบละเอียด, อธิบายเพิ่ม, ยกตัวอย่าง, หรือถามต่อ\n"
        "7. หากผู้ใช้ถามถึงเนื้อหาที่ต้องอาศัยข้อมูลเฉพาะเจาะจงแต่ยังไม่มีการอัปโหลดเอกสารเลยในเซสชันนี้ ให้แนะนำอย่างสุภาพว่าเขาสามารถอัปโหลดไฟล์ (PDF/Text) เพื่อให้คุณช่วยวิเคราะห์เนื้อหานั้นได้อย่างแม่นยำยิ่งขึ้น\n"
        "8. ก่อนตอบ ให้เช็กตัวเองว่าทุกข้อเท็จจริงสำคัญมีหลักฐานจาก Context หรือมาจากความรู้ทั่วไปที่ระบุชัดแล้ว ถ้ายังไม่ชัด ให้ตอบว่าไม่แน่ใจหรือไม่พบข้อมูล แทนการเดา",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "ข้อมูลอ้างอิงจากเซสชันปัจจุบัน:\n{context}\n\nคำถาม: {question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

memory_store = {}
document_sessions = set()
session_uploaded_filenames: Dict[str, List[str]] = {}


def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
        )
    return memory_store[session_id]


def _extract_text_from_uploaded_file(file_bytes: bytes, filename: str) -> str:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("ยังไม่รองรับ PDF เพราะยังไม่ได้ติดตั้ง pypdf") from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    if lower_name.endswith((".txt", ".md", ".csv", ".json")):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    decoded = file_bytes.decode("utf-8", errors="ignore").strip()
    if decoded:
        return decoded

    raise ValueError("รองรับไฟล์ .pdf, .txt, .md, .csv, .json เป็นหลัก")


def _normalize_extracted_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    lines = []
    previous_blank = False
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue

        if lines and lines[-1] and not lines[-1].endswith((".", "?", "!", ":", ";")) and len(lines[-1]) < 140:
            lines[-1] = f"{lines[-1]} {line}"
        else:
            lines.append(line)
        previous_blank = False

    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _looks_like_garbage(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 40:
        return False

    alnum_count = sum(char.isalnum() for char in compact)
    return (alnum_count / max(len(compact), 1)) < 0.45


def _contains_priority_terms(text: str) -> bool:
    lowered = (text or "").lower()
    priority_terms = [
        "อาจารย์",
        "ผู้สอน",
        "ผู้รับผิดชอบรายวิชา",
        "ชื่อ",
        "รายชื่อ",
        "รายวิชา",
        "รหัสวิชา",
        "teacher",
        "lecturer",
        "instructor",
        "course",
        "subject",
        "name",
        "เกรด",
        "คะแนน",
        "ประเมินผล",
        "ตัดเกรด",
        "ผ่าน",
        "ตก",
        "f",
        "grade",
        "grading",
        "score",
        "pass",
        "fail",
    ]
    return any(term in lowered for term in priority_terms)


def _tokenize_for_overlap(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[ก-๙]+", (text or "").lower())


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = _normalize_extracted_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "\n...[truncated]"


def _trim_chat_history(messages, max_messages: int = 4):
    if not messages:
        return []
    return messages[-max_messages:]


def _keyword_overlap_score(question: str, text: str) -> float:
    question_tokens = set(_tokenize_for_overlap(question))
    text_tokens = set(_tokenize_for_overlap(text))
    if not question_tokens or not text_tokens:
        return 0.0

    ignored_tokens = {
        "คือ", "และ", "หรือ", "the", "a", "an", "what", "when", "where",
        "who", "why", "how", "ได้", "ไหม", "อะไร", "ของ", "ที่", "ใน",
    }
    filtered_question_tokens = {
        token for token in question_tokens if len(token) > 1 and token not in ignored_tokens
    }
    if not filtered_question_tokens:
        filtered_question_tokens = question_tokens

    overlap = filtered_question_tokens & text_tokens
    return len(overlap) / max(len(filtered_question_tokens), 1)


def _is_specific_question(question: str) -> bool:
    lowered = (question or "").lower()
    specific_terms = [
        "อาจารย์", "ผู้สอน", "ชื่อ", "รายชื่อ", "รหัสวิชา", "หน่วยกิต",
        "section", "teacher", "lecturer", "instructor", "course code",
        "credits", "room", "schedule", "เวลา", "ห้อง", "ตาราง",
        "เกรด", "คะแนน", "ประเมินผล", "ตัดเกรด", "ผ่าน", "ตก", "ติดf",
        "grade", "grading", "score", "pass", "fail",
    ]
    has_code = bool(re.search(r"[A-Za-z]{2,}\s*-?\d{2,}", question or ""))
    return has_code or any(term in lowered for term in specific_terms)


def _is_grading_question(question: str) -> bool:
    lowered = re.sub(r"\s+", "", (question or "").lower())
    grading_terms = [
        "เกรด", "คะแนน", "ประเมินผล", "ตัดเกรด", "ผ่าน", "ตก", "ติดf",
        "grade", "grading", "score", "pass", "fail",
    ]
    return any(term in lowered for term in grading_terms)


def _is_exam_generation_question(question: str) -> bool:
    lowered = (question or "").lower()
    exam_terms = [
        "ข้อสอบ", "แบบฝึกหัด", "quiz", "exam", "test",
        "generate questions", "ออกข้อสอบ", "เจนข้อสอบ",
    ]
    return any(term in lowered for term in exam_terms)


def _expand_search_queries(question: str) -> List[str]:
    normalized_question = _normalize_extracted_text(question)
    lowered = normalized_question.lower()
    queries = [normalized_question]

    if any(keyword in lowered for keyword in ["อาจารย์", "ผู้สอน", "teacher", "lecturer", "instructor"]):
        queries.append(f"{normalized_question} อาจารย์ผู้สอน ผู้รับผิดชอบรายวิชา")

    if any(keyword in lowered for keyword in ["ชื่อ", "name"]):
        queries.append(f"{normalized_question} รายชื่อ")

    if any(keyword in lowered for keyword in ["วิชา", "course", "subject", "รายวิชา"]):
        queries.append(f"{normalized_question} รายวิชา หมวดที่ 1 ข้อมูลทั่วไป")

    if _is_grading_question(question):
        queries.append(f"{normalized_question} การประเมินผล เกณฑ์คะแนน เกรด ตัดเกรด ผ่าน ตก F")

    return list(dict.fromkeys(query for query in queries if query))


def _chunk_text(text: str) -> List[str]:
    if not text:
        return []

    cleaned_text = _normalize_extracted_text(text)
    if not cleaned_text:
        return []

    pre_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=800,
        chunk_overlap=120,
    )
    pre_chunks = pre_splitter.split_text(cleaned_text)

    deduped_chunks = []
    seen = set()
    for chunk in pre_chunks:
        normalized_chunk = _normalize_extracted_text(chunk)
        is_short = len(normalized_chunk) < 80
        if _looks_like_garbage(normalized_chunk):
            continue
        if is_short and not _contains_priority_terms(normalized_chunk):
            continue
        key = normalized_chunk[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped_chunks.append(normalized_chunk)

    return deduped_chunks


def _clear_namespace(namespace: str):
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass


def _upsert_chunks_to_pinecone(chunks: List[str], namespace: str):
    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )


def _search_pinecone(question: str, namespace: str, k: int = 4) -> str:
    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=namespace,
    )

    docs = []
    seen = set()
    for query in _expand_search_queries(question):
        try:
            matched_docs = vector_store.similarity_search(query, k=k)
        except Exception:
            matched_docs = []

        for doc in matched_docs:
            content = doc.page_content.strip()
            if content and content not in seen:
                seen.add(content)
                docs.append(doc)

    if not docs:
        return ""

    selected_chunks = []
    total_length = 0
    for doc in docs:
        chunk_text = _normalize_extracted_text(doc.page_content)
        if not chunk_text or _looks_like_garbage(chunk_text):
            continue
        if len(chunk_text) < 80 and not _contains_priority_terms(chunk_text):
            continue
        if _is_specific_question(question):
            overlap_score = _keyword_overlap_score(question, chunk_text)
            if overlap_score < 0.15 and not _contains_priority_terms(chunk_text):
                continue

        if selected_chunks and total_length + len(chunk_text) > 3200:
            break

        selected_chunks.append(chunk_text)
        total_length += len(chunk_text)

    return "\n\n".join(selected_chunks)


def _search_pinecone_chunks(question: str, namespace: str, k: int = 4) -> List[str]:
    context_text = _search_pinecone(question, namespace, k=k)
    if not context_text:
        return []
    return [chunk.strip() for chunk in context_text.split("\n\n") if chunk.strip()]


def _build_references(chunks: List[str], filenames: Optional[List[str]] = None) -> List[dict]:
    references = []
    source_name = ", ".join(filenames or []) if filenames else "เอกสารที่อัปโหลด"
    for chunk in chunks[:3]:
        snippet = _truncate_text(chunk, 220)
        if snippet:
            references.append({
                "source": source_name,
                "snippet": snippet,
            })
    return references


def _should_refuse_answer(question: str, context_text: str, has_uploaded_document: bool) -> bool:
    if not _is_specific_question(question):
        return False

    normalized_context = _normalize_extracted_text(context_text)
    if not normalized_context:
        return has_uploaded_document

    refusal_markers = [
        "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัปโหลดสำหรับคำถามนี้",
        "ไม่มีข้อมูลจากเอกสารอ้างอิง",
    ]
    if any(marker in normalized_context for marker in refusal_markers):
        return True

    if _is_grading_question(question):
        has_grading_evidence = any(
            term in normalized_context.lower()
            for term in ["เกรด", "คะแนน", "ประเมินผล", "ตัดเกรด", "grade", "grading", "score", "pass", "fail", "f"]
        )
        has_numeric_evidence = bool(re.search(r"\d+", normalized_context))
        if not (has_grading_evidence and has_numeric_evidence):
            return True

    return _keyword_overlap_score(question, normalized_context) < 0.12


def ask_chatbot(
    session_id: str,
    question: str,
    uploaded_files: Optional[List[dict]] = None,
) -> dict:
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({})["chat_history"]
    namespace = session_id
    warning = None

    if uploaded_files:
        all_new_chunks = []
        uploaded_names = []
        for file_data in uploaded_files:
            file_bytes = file_data.get("bytes")
            filename = file_data.get("filename")

            if file_bytes and filename:
                uploaded_names.append(filename)
                text = _extract_text_from_uploaded_file(file_bytes, filename)
                if text:
                    all_new_chunks.extend(_chunk_text(text))

        if all_new_chunks:
            _clear_namespace(namespace)
            _upsert_chunks_to_pinecone(all_new_chunks, namespace)
            document_sessions.add(session_id)
            session_uploaded_filenames[session_id] = uploaded_names

    context_chunks = _search_pinecone_chunks(question, namespace)
    context_text = "\n\n".join(context_chunks)
    if not context_chunks:
        if session_id in document_sessions:
            context_text = (
                "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่อัปโหลดสำหรับคำถามนี้ "
                "(ห้ามเดารายละเอียดที่ไม่มีหลักฐานจากเอกสาร)"
            )
            warning = "ไม่พบข้อมูลที่ชัดเจนพอในเอกสารสำหรับคำถามนี้"
        else:
            context_text = "ไม่มีข้อมูลจากเอกสารอ้างอิง (ผู้ใช้ยังไม่ได้อัปโหลดไฟล์ในเซสชันนี้)"
            warning = "ยังไม่มีเอกสารในเซสชันนี้ ถ้าต้องการคำตอบจากไฟล์ แนะนำให้อัปโหลดเอกสารก่อน"

    if _should_refuse_answer(question, context_text, session_id in document_sessions):
        response = "ขอไม่ตอบแบบเดานะครับ เพราะไม่พบข้อมูลที่ชัดเจนพอในเอกสารที่อัปโหลด"
        memory.save_context(
            {"input": question},
            {"output": response},
        )
        return {
            "answer": response,
            "references": _build_references(context_chunks, session_uploaded_filenames.get(session_id)),
            "warning": "คำถามนี้ต้องอาศัยข้อมูลเฉพาะ แต่เอกสารยังไม่มีหลักฐานชัดเจนพอ",
        }

    if _is_exam_generation_question(question):
        question = (
            f"{question}\n\n"
            "ข้อกำหนดเพิ่มเติม:\n"
            "- ถ้าผู้ใช้ไม่ได้ระบุจำนวนข้อ ให้สร้างไม่เกิน 5 ข้อ\n"
            "- คำอธิบายและเฉลยให้สั้น กระชับ\n"
            "- ห้ามสร้างเนื้อหายาวเกินจำเป็น"
        )
        context_text = _truncate_text(context_text, 1200)
        question = _truncate_text(question, 500)
        chat_history = _trim_chat_history(chat_history, 2)
        if len(context_chunks) >= 3:
            warning = "หากต้องการข้อสอบแม่นขึ้น ลองระบุบทหรือหัวข้อที่ต้องการออกข้อสอบ"
    else:
        context_text = _truncate_text(context_text, 2200)
        question = _truncate_text(question, 700)
        chat_history = _trim_chat_history(chat_history, 4)

    response = chain.invoke(
        {
            "chat_history": chat_history,
            "context": context_text,
            "question": question,
        }
    )

    memory.save_context(
        {"input": question},
        {"output": response},
    )

    return {
        "answer": response,
        "references": _build_references(context_chunks, session_uploaded_filenames.get(session_id)),
        "warning": warning,
    }
