import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import MessagesPlaceholder
from langchain_classic.memory import ConversationBufferWindowMemory

# โหลด Environment Variables
load_dotenv()

# การระบุ Path ของโฟลเดอร์ faiss_index ให้สัมพันธ์กับไฟล์ปัจจุบัน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

# โหลด Embeddings และ FAISS DB ให้ใช้แพ็กเกจใหม่เพื่อแก้แจ้งเตือน Warning
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    encode_kwargs={"normalize_embeddings": True}
)

new_db = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = new_db.as_retriever(search_kwargs={"k": 3})

# ตั้งค่า LLM (Typhoon) เป็น temp=0.0 เพื่อความแม่นยำสูงสุด ห้ามแต่งเองเด็ดขาด
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_KEY"),
    model='typhoon-v2.5-30b-a3b-instruct',
    temperature=0.0,
    max_tokens=8000
)

# ตั้งค่า Prompt รวม MessagesPlaceholder เพื่อเก็บ History
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "คุณคือ 'Learnify Bot' ผู้ช่วยและที่ปรึกษาด้านวิชาการ หน้าที่หลักของคุณคือการอ่านและวิเคราะห์เอกสารประกอบการสอน (Course Syllabus / OBE3) หรือเอกสารรายวิชาที่แนบมาให้ เพื่อตอบคำถามและให้ความรู้แก่นักศึกษาในรายวิชานั้นๆ\n\n"
     "ข้อควรปฏิบัติ:\n"
     "1. ตอบคำถามและให้คำอธิบายโดยอิงจาก 'ข้อมูลอ้างอิง (Context)' ที่มาจากเอกสารรายวิชาเป็นหลัก\n"
     "2. หากนักศึกษาถามถึงเนื้อหาหรือเรื่องที่ไม่มีในผลลัพธ์การค้นหา ให้ตอบตามความเป็นจริงว่าไม่พบข้อมูลนั้นในเอกสารรายวิชาปัจจุบัน\n"
     "3. ให้คำแนะนำเกี่ยวกับการเรียนการสอน การเตรียมตัวสอบ หรือจุดประสงค์รายวิชาได้อย่างเป็นมิตรและเข้าใจง่ายในฐานะอาจารย์ที่ปรึกษา\n\n"
     "จำไว้: ให้ความรู้ที่เกี่ยวข้องกับเอกสารเป็นหลัก ห้ามแต่งชื่อวิชาหรือรหัสประจำวิชาขึ้นมาเองถ้าไม่มีในข้อมูล"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}")
])

parser = StrOutputParser()

# เชื่อม Chain หลักแบบไม่ต้องใช้ RunnableWithMessageHistory (เน้นเข้าใจง่าย)
chain = prompt | llm | parser

# จำลองหน่วยความจำในเครื่อง เพื่อใช้เก็บ Chat History แบบแยกตามคน
store = {}

def get_session_memory(session_id: str) -> ConversationBufferWindowMemory:
    """ฟังก์ชันที่ใช้ดึง Buffer Memory ของผู้ใช้ (จำกัดปริมาณการจำเพื่อไม่ให้ Token ล้น)"""
    if session_id not in store:
        # กำหนด k=10 หมายถึง เก็บ 10 บทสนทนาไป-กลับ ล่าสุด เสมือนจำได้ตลอดแต่ไม่กินโหลดเซิร์ฟ
        store[session_id] = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history", 
            return_messages=True
        )
    return store[session_id]

def ask_chatbot(session_id: str, question: str) -> str:
    """
    ฟังก์ชันหลักที่ให้ Endpoint เรียกใช้งาน โดยดึง memory แล้ว invoke โดยตรงตามตัวอย่างที่เข้าใจง่าย
    """
    # 1. โหลดประวัติแชท (Memory) ของ session นี้จาก Buffer Memory
    memory = get_session_memory(session_id)
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # 2. ค้นหาเอกสารอ้างอิง (RAG Context)
    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. สั่งให้ Chain ตอบคำถามโดยโยน Input เข้าไปตรงๆ เป็น Dictionary
    response = chain.invoke({
        "chat_history": chat_history,
        "context": context_text,
        "question": question
    })
    
    # 4. บันทึกคำถามของ User และคำตอบของ AI รอบนี้ลง Memory แบบ Manual
    memory.save_context(
        {"input": question}, 
        {"output": response}
    )
    
    return response
