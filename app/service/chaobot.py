import io
import os
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


def _chunk_text(text: str) -> List[str]:
    if not text:
        return []

    # 1. Pre-split ด้วยขนาดปานกลาง เพื่อกัน OOM หากเอกสารมีขนาดใหญ่เกินไป
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=0, # ไม่ทำ overlap ตรงนี้ เพราะเดี๋ยว SemanticChunker จะเช็คความสัมพันธ์เอง
    )
    pre_chunks = pre_splitter.split_text(text)

    # 2. ทำ Semantic Chunking ควบคู่ด้วย เพื่อรักษาความหมายของกระโยคไม่ให้ขาดตอน (ลดการหลอน)
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )

    final_chunks = []
    for chunk in pre_chunks:
        try:
            # หั่นแบบรักษากลุ่มความหมาย
            semantic_chunks = semantic_splitter.split_text(chunk)
            final_chunks.extend(semantic_chunks)
        except Exception:
            # Fallback เผื่อเจอ Error ในบางก้อน
            final_chunks.append(chunk)

    return final_chunks if final_chunks else pre_chunks


# ────────────────────────────────────────────────────────────
# Pinecone Helpers
# ────────────────────────────────────────────────────────────
def _clear_namespace(namespace: str):
    """ลบ vectors ทั้งหมดใน namespace นั้น (= ลบเอกสารเก่าของ user)"""
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
    except Exception:
        # namespace อาจยังไม่เคยมี data → ไม่ต้อง error
        pass


def _upsert_chunks_to_pinecone(chunks: List[str], namespace: str):
    """อัปโหลด chunks เข้า Pinecone ใน namespace ของ user"""
    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace,
    )


def _search_pinecone(question: str, namespace: str, k: int = 4) -> str:
    """ค้นหา chunks ที่เกี่ยวข้องจาก Pinecone"""
    vector_store = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=namespace,
    )

    docs = vector_store.similarity_search(question, k=k)
    if not docs:
        return ""

    context_text = "\n\n".join([doc.page_content for doc in docs])
    # Trim ไม่ให้ context ยาวเกินไป
    if len(context_text) > 3000:
        context_text = context_text[:3000] + "\n...(ตัดทอนเนื้อหาบางส่วน)"
    return context_text


def _check_namespace_has_data(namespace: str) -> bool:
    """เช็คว่า namespace นี้มี vectors อยู่หรือไม่"""
    try:
        stats = pinecone_index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        ns_info = namespaces.get(namespace, {})
        return ns_info.get("vector_count", 0) > 0
    except Exception:
        return False


# ────────────────────────────────────────────────────────────
# Main Function
# ────────────────────────────────────────────────────────────
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
