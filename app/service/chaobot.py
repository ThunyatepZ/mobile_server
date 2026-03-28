import io
import os
import re
from typing import List, Optional

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
        "1. หากผู้ใช้อัปโหลดเอกสารมา (ดูจาก Context) ให้เน้นตอบโดยอิงจากข้อมูลในเอกสารนั้นเป็นหลัก\n"
        "2. หากข้อมูลที่ถามไม่มีในเอกสาร หรือผู้ใช้ยังไม่ได้อัปโหลดเอกสาร คุณสามารถตอบโดยใช้ความรู้ทั่วไปที่คุณมีได้ตามความเหมาะสม แต่ควรแจ้งให้ผู้ใช้ทราบหากข้อมูลนั้นไม่ได้มาจากเอกสารที่เขาให้มา\n"
        "3. ให้คำแนะนำด้วยน้ำเสียงที่เป็นมิตร กระตือรือร้น และส่งเสริมการเรียนรู้\n"
        "4. ให้ตอบแบบสรุปสั้น กระชับ เป็นค่าเริ่มต้น โดยปกติไม่เกิน 3-5 บรรทัด หรือ 3 bullet สั้นๆ\n"
        "5. ถ้าผู้ใช้ถามหาข้อมูลเฉพาะ เช่น ชื่ออาจารย์ รหัสวิชา จำนวนหน่วยกิต ให้ตอบเป็นคำตอบตรงๆก่อน ไม่ต้องเกริ่นยาว\n"
        "6. ให้ลงรายละเอียดเพิ่มเติมเฉพาะเมื่อผู้ใช้ขอ เช่น ขอแบบละเอียด, อธิบายเพิ่ม, ยกตัวอย่าง, หรือถามต่อ\n"
        "7. หากผู้ใช้ถามถึงเนื้อหาที่ต้องอาศัยข้อมูลเฉพาะเจาะจงแต่ยังไม่มีการอัปโหลดเอกสารเลยในเซสชันนี้ ให้แนะนำอย่างสุภาพว่าเขาสามารถอัปโหลดไฟล์ (PDF/Text) เพื่อให้คุณช่วยวิเคราะห์เนื้อหานั้นได้อย่างแม่นยำยิ่งขึ้น",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "ข้อมูลอ้างอิงจากเซสชันปัจจุบัน:\n{context}\n\nคำถาม: {question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

memory_store = {}


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
    ]
    return any(term in lowered for term in priority_terms)


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

    return list(dict.fromkeys(query for query in queries if query))


def _chunk_text(text: str) -> List[str]:
    if not text:
        return []

    cleaned_text = _normalize_extracted_text(text)
    if not cleaned_text:
        return []

    pre_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1400,
        chunk_overlap=180,
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

        if selected_chunks and total_length + len(chunk_text) > 3200:
            break

        selected_chunks.append(chunk_text)
        total_length += len(chunk_text)

    return "\n\n".join(selected_chunks)


def ask_chatbot(
    session_id: str,
    question: str,
    uploaded_files: Optional[List[dict]] = None,
) -> str:
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({})["chat_history"]
    namespace = session_id

    if uploaded_files:
        all_new_chunks = []
        for file_data in uploaded_files:
            file_bytes = file_data.get("bytes")
            filename = file_data.get("filename")

            if file_bytes and filename:
                text = _extract_text_from_uploaded_file(file_bytes, filename)
                if text:
                    all_new_chunks.extend(_chunk_text(text))

        if all_new_chunks:
            _clear_namespace(namespace)
            _upsert_chunks_to_pinecone(all_new_chunks, namespace)

    context_text = _search_pinecone(question, namespace)
    if not context_text:
        context_text = "ไม่มีข้อมูลจากเอกสารอ้างอิง (ผู้ใช้ยังไม่ได้อัปโหลดไฟล์ในเซสชันนี้)"

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

    return response
