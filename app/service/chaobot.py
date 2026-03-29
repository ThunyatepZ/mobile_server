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
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# โหลด Environment Variables
load_dotenv()

# ────────────────────────────────────────────────────────────
# Embeddings  (BAAI/bge-m3 → dimension 1024)
# ────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

# ────────────────────────────────────────────────────────────
# Pinecone
# ────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "learnify-docs"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# สร้าง index ถ้ายังไม่มี
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,  # ตรงกับ BAAI/bge-m3
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# ────────────────────────────────────────────────────────────
# LLM (Typhoon)
# ────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_KEY"),
    model="typhoon-v2.5-30b-a3b-instruct",
    temperature=0.0,
    max_tokens=8192,
)

# ────────────────────────────────────────────────────────────
# Prompt
# ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณคือ 'Learnify Bot' ผู้ช่วยส่วนตัวที่ฉลาดและเป็นมิตร หน้าที่ของคุณคือช่วยผู้ใช้เรียนรู้และทำความเข้าใจเนื้อหาต่างๆ\n\n"
        "ข้อควรปฏิบัติ:\n"
        "1. หากผู้ใช้อัปโหลดเอกสารมา (ดูจาก Context) ให้เน้นตอบโดยอิงจากข้อมูลในเอกสารนั้นเป็นหลัก และให้คำอธิบายที่ละเอียดและเข้าใจง่าย\n"
        "2. หากข้อมูลที่ถามไม่มีในเอกสาร หรือผู้ใช้ยังไม่ได้อัปโหลดเอกสาร คุณสามารถตอบโดยใช้ความรู้ทั่วไปที่คุณมีได้ตามความเหมาะสม แต่ควรแจ้งให้ผู้ทราบหากข้อมูลนั้นไม่ได้มาจากเอกสารที่เขาให้มา\n"
        "3. ให้คำแนะนำด้วยน้ำเสียงที่เป็นมิตร กระตือรือร้น และส่งเสริมการเรียนรู้\n"
        "4. หากผู้ใช้ถามถึงเนื้อหาที่ต้องอาศัยข้อมูลเฉพาะเจาะจงแต่ 'ยังไม่มีการอัปโหลดเอกสารเลยในเซสชันนี้' ให้แนะนำอย่างสุภาพว่าเขาสามารถอัปโหลดไฟล์ (PDF/Text) เพื่อให้คุณช่วยวิเคราะห์เนื้อหานั้นได้อย่างแม่นยำยิ่งขึ้น",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "ข้อมูลอ้างอิงจากเซสชันปัจจุบัน:\n{context}\n\nคำถาม: {question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

# Chat History (in-memory แยกตาม session)
memory_store = {}


def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """ดึง Chat Memory ของผู้ใช้ (เก็บ 10 รอบสนทนาล่าสุด)"""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
        )
    return memory_store[session_id]


# ────────────────────────────────────────────────────────────
# File Processing Helpers
# ────────────────────────────────────────────────────────────
def _extract_text_from_uploaded_file(file_bytes: bytes, filename: str) -> str:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("ยังไม่รองรับ PDF เพราะยังไม่ได้ติดตั้ง pypdf") from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if lower_name.endswith((".txt", ".md", ".csv", ".json")):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    decoded = file_bytes.decode("utf-8", errors="ignore").strip()
    if decoded:
        return decoded

    raise ValueError("รองรับไฟล์ .pdf, .txt, .md, .csv, .json เป็นหลัก")

# def _chunk_text(text: str) -> List[str]:
#     """แบ่ง Chunk ข้อความโดยใช้ RecursiveCharacterTextSplitter เพื่อคุณภาพที่ดีขึ้น"""
#     if not text:
#         return []
    
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     return splitter.split_text(text)

def _chunk_text(text: str) -> List[str]:
    """
    Semantic chunk แบบไม่ไปแตะส่วนอื่นของระบบ
    แนวคิด:
    1) แบ่งเอกสารเป็น paragraph / section ก่อน
    2) รวม paragraph ที่มีความต่อเนื่องกันจนกว่าจะใกล้เต็ม chunk
    3) ถ้า paragraph ไหนยาวเกินไป ค่อย split ย่อยด้วย sentence
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # ปรับ whitespace ให้สะอาดขึ้น แต่ยังคง \n\n สำหรับแบ่งย่อหน้า
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    target_chunk_size = 1000
    max_chunk_size = 1200
    min_chunk_size = 300

    # แบ่งแบบ semantic เบื้องต้นด้วย paragraph
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current_chunk, current_len
        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = []
            current_len = 0

    def split_long_paragraph(paragraph: str) -> List[str]:
        """
        ถ้าย่อหน้ายาวเกินไป ให้ split ตาม sentence ก่อน
        ถ้ายังยาวอีก ค่อย fallback เป็น character splitter
        """
        if len(paragraph) <= max_chunk_size:
            return [paragraph]

        # แยก sentence ไทย/อังกฤษแบบง่าย ๆ
        sentences = re.split(r"(?<=[\.\!\?\n])\s+|(?<=।)\s+", paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=target_chunk_size,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            return fallback_splitter.split_text(paragraph)

        parts: List[str] = []
        buffer: List[str] = []
        buffer_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if buffer and buffer_len + sent_len + 1 > max_chunk_size:
                parts.append(" ".join(buffer).strip())
                buffer = [sent]
                buffer_len = sent_len
            else:
                buffer.append(sent)
                buffer_len += sent_len + (1 if buffer else 0)

        if buffer:
            parts.append(" ".join(buffer).strip())

        # ถ้ายังมี part ที่ยาวเกินอีก ค่อย fallback
        final_parts: List[str] = []
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chunk_size,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        for part in parts:
            if len(part) > max_chunk_size:
                final_parts.extend(fallback_splitter.split_text(part))
            else:
                final_parts.append(part)

        return final_parts

    for para in paragraphs:
        para_parts = split_long_paragraph(para)

        for part in para_parts:
            part_len = len(part)

            # ถ้ายังพอรวมใน chunk เดิมได้ ก็รวม
            if current_chunk and current_len + part_len + 2 <= target_chunk_size:
                current_chunk.append(part)
                current_len += part_len + 2
                continue

            # ถ้า chunk เดิมมีขนาดพอแล้ว ค่อยปิด chunk
            if current_chunk and current_len >= min_chunk_size:
                flush_current()

            # เริ่ม chunk ใหม่
            current_chunk.append(part)
            current_len = part_len

            # ถ้ายาวมาก ก็ปิดทันที
            if current_len >= max_chunk_size:
                flush_current()

    flush_current()

    # กันกรณี chunk ท้ายสั้นมาก ให้ไปรวมกับอันก่อน
    merged_chunks: List[str] = []
    for chunk in chunks:
        if merged_chunks and len(chunk) < 150:
            merged_chunks[-1] = merged_chunks[-1].rstrip() + "\n\n" + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks


def _build_context_from_uploaded_file(question: str, file_bytes: bytes, filename: str) -> str:
    text = _extract_text_from_uploaded_file(file_bytes, filename)
    if not text:
        raise ValueError("ไม่สามารถอ่านข้อความจากไฟล์ที่อัปโหลดได้")

    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("ไม่พบเนื้อหาที่ใช้สร้าง context จากไฟล์นี้")

    # สร้าง Vector Store ชั่วคราวจากไฟล์ที่อัปโหลด
    uploaded_db = FAISS.from_texts(chunks, embedding=embeddings)
    docs = uploaded_db.similarity_search(question, k=min(4, len(chunks)))
    return "\n\n".join([doc.page_content for doc in docs])


def ask_chatbot(
    session_id: str,
    question: str,
    uploaded_files: Optional[List[dict]] = None,
) -> str:
    """
    ฟังก์ชันหลักที่ให้ Endpoint เรียกใช้งาน
    - ถ้ามีไฟล์อัปโหลดมา → ลบ chunks เดิม → เพิ่ม chunks ใหม่ใน Pinecone
    - ถ้าไม่มีไฟล์ → ใช้ chunks เดิมที่อยู่ใน Pinecone
    """
    # 1. โหลด Chat History
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({})["chat_history"]

    # namespace = email ของ user (session_id)
    namespace = session_id

    # 2. จัดการไฟล์อัปโหลด
    if uploaded_files:
        all_new_chunks = []
        for file_data in uploaded_files:
            file_bytes = file_data.get("bytes")
            filename = file_data.get("filename")

            if file_bytes and filename:
                text = _extract_text_from_uploaded_file(file_bytes, filename)
                if text:
                    chunks = _chunk_text(text)
                    all_new_chunks.extend(chunks)

        if all_new_chunks:
            # ลบ chunks เก่าทิ้ง → เพิ่ม chunks ใหม่ทั้งหมด
            _clear_namespace(namespace)
            _upsert_chunks_to_pinecone(all_new_chunks, namespace)

    # 3. ค้นหา context จาก Pinecone
    if _check_namespace_has_data(namespace):
        context_text = _search_pinecone(question, namespace)
    else:
        context_text = "ไม่มีข้อมูลจากเอกสารอ้างอิง (ผู้ใช้ยังไม่ได้อัปโหลดไฟล์ในเซสชันนี้)"

    # 4. สั่งให้ Chain ตอบคำถาม
    response = chain.invoke(
        {
            "chat_history": chat_history,
            "context": context_text,
            "question": question,
        }
    )

    # 5. บันทึก Chat History
    memory.save_context(
        {"input": question},
        {"output": response},
    )

    return response
