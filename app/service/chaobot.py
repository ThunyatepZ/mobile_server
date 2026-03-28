import io
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker

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

    decoded = file_bytes.decode("utf-8", errors="ignore").strip()
    if decoded:
        return decoded

    raise ValueError("รองรับไฟล์ .pdf, .txt, .md, .csv, .json เป็นหลัก")


def _chunk_text(text: str) -> List[str]:
    if not text:
        return []
    
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )
    return splitter.split_text(text)


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
    สามารถจำเอกสารที่เคยอัปโหลดไว้ก่อนหน้าใน Session เดียวกันได้
    """
    # 1. โหลดข้อมูล Session
    session_data = get_session_data(session_id)
    memory = session_data["memory"]
    chat_history = memory.load_memory_variables({})["chat_history"]

    # 2. จัดการไฟล์อัปโหลด (ถ้ามีส่งมาใหม่ ให้เพิ่มเข้าไปใน Vector Store เดิม)
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
            # ถ้ามี Vector Store เดิมอยู่แล้ว ให้เพิ่ม Chunks ใหม่เข้าไป
            if session_data["vector_store"] is not None:
                session_data["vector_store"].add_texts(all_new_chunks)
            else:
                # ถ้ายังไม่มี ให้สร้างใหม่
                session_data["vector_store"] = FAISS.from_texts(all_new_chunks, embedding=embeddings)

    # 3. ค้นหาเอกสารอ้างอิงจาก Vector Store ที่อยู่ใน Session
    vector_store = session_data.get("vector_store")
    if vector_store:
        # ค้นหาข้อมูลที่ใกล้เคียงที่สุด 4 ส่วน
        docs = vector_store.similarity_search(question, k=4)
        context_text = "\n\n".join([doc.page_content for doc in docs])
    else:
        # กรณีไม่มีเอกสารอัปโหลดเลย ทั้งในรอบนี้และรอบก่อนๆ
        context_text = "ไม่มีข้อมูลจากเอกสารอ้างอิง (ผู้ใช้ยังไม่ได้อัปโหลดไฟล์ในเซสชันนี้)"

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
