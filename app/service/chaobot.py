import io
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# โหลด Environment Variables
load_dotenv()

# โหลด Embeddings ให้ใช้แพ็กเกจใหม่เพื่อแก้แจ้งเตือน Warning
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

# ตั้งค่า LLM (Typhoon) เป็น temp=0.0 เพื่อความแม่นยำสูงสุด
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_KEY"),
    model="typhoon-v2.5-30b-a3b-instruct",
    temperature=0.0,
    max_tokens=8000,
)

# ตั้งค่า Prompt รวม MessagesPlaceholder เพื่อเก็บ History
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "คุณคือ 'Learnify Bot' ผู้ช่วยส่วนตัวของผู้ใช้ หน้าที่หลักของคุณคือการอ่านและวิเคราะห์เอกสารที่ผู้ใช้อัปโหลดมาให้\n\n"
        "ข้อควรปฏิบัติ:\n"
        "1. ตอบคำถามและให้คำอธิบายโดยอิงจาก 'ข้อมูลอ้างอิง (Context)' ที่มาจากเอกสารที่ผู้ใช้อัปโหลดเท่านั้น\n"
        "2. หากข้อมูลที่ถามไม่มีในเอกสาร ให้ตอบตามความเป็นจริงว่าไม่พบข้อมูลนั้นในเอกสารที่ให้มา\n"
        "3. ให้คำแนะนำด้วยน้ำเสียงที่เป็นมิตรและเข้าใจง่าย\n"
        "4. หากไม่มีข้อมูลอ้างอิง (Context) จากเอกสาร ให้แจ้งผู้ใช้ว่ากรุณาอัปโหลดเอกสารก่อนถามคำถามเกี่ยวกับเนื้อหา",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}"),
])

parser = StrOutputParser()

# เชื่อม Chain หลัก
chain = prompt | llm | parser

# จำลองหน่วยความจำในเครื่อง เพื่อใช้เก็บ Chat History แบบแยกตามคน
store = {}


def get_session_data(session_id: str):
    """ฟังก์ชันที่ใช้ดึงข้อมูล Session (Memory + VectorDB) ของผู้ใช้"""
    if session_id not in store:
        store[session_id] = {
            "memory": ConversationBufferWindowMemory(
                k=10,
                memory_key="chat_history",
                return_messages=True,
            ),
            "vector_store": None
        }
    return store[session_id]


def _extract_text_from_uploaded_file(file_bytes: bytes, filename: str) -> str:
    lower_name = (filename or "").lower()

    if lower_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("ยังไม่รองรับ PDF เพราะยังไม่ได้ติดตั้ง pypdf (ลอง pip install pypdf)") from exc

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if lower_name.endswith((".txt", ".md", ".csv", ".json")):
        return file_bytes.decode("utf-8", errors="ignore").strip()

    # fallback
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
    uploaded_file_bytes: Optional[bytes] = None,
    uploaded_filename: Optional[str] = None,
) -> str:
    """
    ฟังก์ชันหลักที่ให้ Endpoint เรียกใช้งาน
    สามารถจำเอกสารที่เคยอัปโหลดไว้ก่อนหน้าใน Session เดียวกันได้
    """
    # 1. โหลดข้อมูล Session
    session_data = get_session_data(session_id)
    memory = session_data["memory"]
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 2. จัดการไฟล์อัปโหลด (ถ้ามีส่งมาใหม่ ให้สร้าง Vector Store ชุดใหม่ทับของเดิม)
    if uploaded_file_bytes and uploaded_filename:
        text = _extract_text_from_uploaded_file(uploaded_file_bytes, uploaded_filename)
        if text:
            chunks = _chunk_text(text)
            if chunks:
                # สร้างและเก็บ Vector Store ไว้ใน Session
                session_data["vector_store"] = FAISS.from_texts(chunks, embedding=embeddings)
            else:
                raise ValueError("ไม่พบเนื้อหาที่แบ่งเป็นส่วนๆ ได้ในไฟล์นี้")
        else:
            raise ValueError("ไม่สามารถอ่านข้อความจากไฟล์ที่อัปโหลดได้")

    # 3. ค้นหาเอกสารอ้างอิงจาก Vector Store ที่อยู่ใน Session
    vector_store = session_data.get("vector_store")
    if vector_store:
        # ค้นหาข้อมูลที่ใกล้เคียงที่สุด 4 ส่วน
        docs = vector_store.similarity_search(question, k=4)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    else:
        # กรณีไม่มีเอกสารอัปโหลดเลย ทั้งในรอบนี้และรอบก่อนๆ
        context_text = "ไม่พบข้อมูลอ้างอิง เนื่องจากไม่ได้มีการอัปโหลดเอกสาร"

    # 4. สั่งให้ Chain ตอบคำถาม
    response = chain.invoke(
        {
            "chat_history": chat_history,
            "context": context_text,
            "question": question,
        }
    )

    # 5. บันทึกคำถามของ User และคำตอบของ AI ลง Memory
    memory.save_context(
        {"input": question},
        {"output": response},
    )

    return response
