import os
import json
import re
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Setup LLM
llm = ChatOpenAI(
    base_url="https://api.opentyphoon.ai/v1",
    api_key=os.getenv("TYPHOON_KEY"),
    model='typhoon-v2.5-30b-a3b-instruct',
    temperature=0.3,
    max_tokens=8000
)

class QuestionSchema(BaseModel):
    question_text: str = Field(description="The question prompt")
    options: List[str] = Field(description="List of 4 multiple choice options")
    correct_answer: str = Field(description="The exact text of the correct option")
    explanation: str = Field(description="A brief explanation of why this answer is correct")

class QuizSchema(BaseModel):
    title: str = Field(description="A catchy title for the quiz based on content")
    description: str = Field(description="A short summary of what this quiz covers")
    questions: List[QuestionSchema] = Field(description="List of generated questions")

parser = JsonOutputParser(pydantic_object=QuizSchema)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "คุณคืออาจารย์ผู้เชี่ยวชาญด้านการออกข้อสอบ หน้าที่ของคุณคืออ่านเนื้อหาที่ได้รับแล้วสร้างข้อสอบตามที่ผู้ใช้กำหนด\n"
     "ข้อกำหนด:\n"
     "1. สร้างคำถามที่วัดความเข้าใจ ไม่ใช่แค่การจำ\n"
     "2. ให้คำอธิบาย (explanation) ที่ชัดเจนสำหรับแต่ละข้อ\n"
     "3. ผลลัพธ์ต้องเป็นรูปแบบ JSON ตามที่กำหนดเท่านั้น\n"
     "4. ใช้ภาษาไทยในการออกข้อสอบ\n"
     "5. อ่านข้อมูลจากไฟล์PDFที่ได้รับแล้วสร้างข้อสอบ\n"
     "6. หากไม่ทราบข้อมูลควรบอกเชิงความหมายว่าไม่สามารถบอกรายละเอียดนั้นได้ เช่น หากไม่รู้ชื่อผู้สอน ควรเขียนว่าไม่ทราบผู้สอน\n"
     "7. ห้ามพิมพ์ข้อมูลที่ไม่มีในเนื้อหา ยกเว้นข้อมูลที่จำเป็นต่อการออกข้อสอบ\n"
     "8. ห้ามสร้างคำถามที่ซ้ำกัน\n"
     "9. ใช้ภาษาไทยอย่างถูกต้องตามหลักภาษาไทย\n"
     ),
    ("human", "การตั้งค่า:\n{requirements}\n\nเนื้อหาสำหรับออกข้อสอบ:\n{context}\n\n{format_instructions}")
])

quiz_chain = prompt | llm | parser

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted.strip():
            pages.append(extracted)
    return "\n".join(pages).strip()

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _has_enough_text_for_quiz(text: str) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 500:
        return False

    words = normalized.split()
    return len(words) >= 80


def generate_quiz_from_text(
    text: str,
    *,
    num_questions: int = 5,
    difficulty: str = "ปานกลาง",
    question_type: str = "ปรนัย",
    include_explanations: bool = True,
    focus_topic: str = "",
) -> dict:
    safe_num_questions = max(3, min(num_questions, 10))
    normalized_text = _normalize_text(text)
    if not normalized_text:
        raise ValueError("ไม่สามารถอ่านข้อความจากไฟล์ที่อัปโหลดได้ หรือไฟล์ไม่มีข้อความที่ใช้สร้างข้อสอบ")
    if not _has_enough_text_for_quiz(normalized_text):
        raise ValueError(
            "ไฟล์นี้อ่านข้อความได้ไม่พอสำหรับสร้างข้อสอบ อาจเป็น PDF สแกนหรือไฟล์ที่เลือกข้อความไม่ได้ กรุณาใช้ PDF ที่เลือกข้อความได้หรือแปลงเป็นไฟล์ .txt"
        )
    truncated_text = normalized_text[:6000]
    requirements = (
        f"- จำนวนข้อ: {safe_num_questions} ข้อ\n"
        f"- ระดับความยาก: {difficulty}\n"
        f"- ประเภทข้อสอบ: {question_type}\n"
        f"- {'ให้มีเฉลยอธิบายสั้นๆ' if include_explanations else 'ไม่ต้องใส่คำอธิบายเฉลยยาว'}\n"
        f"- เน้นหัวข้อ: {focus_topic or 'ครอบคลุมจากเนื้อหาที่ให้มา'}\n"
        "- ถ้าเนื้อหากว้างเกินไป ให้เลือกประเด็นสำคัญที่สุดก่อน\n"
        "- คำถามทุกข้อต้องอิงจากเนื้อหาที่ให้มาเท่านั้น"
    )
    
    payload = {
        "context": truncated_text,
        "requirements": requirements,
        "format_instructions": parser.get_format_instructions()
    }

    try:
        response = quiz_chain.invoke(payload)
    except OutputParserException as exc:
        raise ValueError(
            "โมเดลสร้างข้อสอบสำเร็จไม่ครบรูปแบบ JSON ที่ระบบต้องการ กรุณาลองใหม่หรือใช้ไฟล์ที่สั้นลง"
        ) from exc
    except Exception as exc:
        raise ValueError(f"การสร้างข้อสอบล้มเหลว: {exc}") from exc

    if not isinstance(response, dict):
        raise ValueError("ระบบสร้างข้อสอบได้ผลลัพธ์ผิดรูปแบบ")

    questions = response.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("ระบบไม่สามารถสร้างรายการคำถามจากไฟล์นี้ได้")

    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("ข้อมูลคำถามที่สร้างได้ผิดรูปแบบ")
        options = question.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError("ตัวเลือกคำตอบของข้อสอบไม่ครบ")

    return response

def save_quiz_to_db(conn, user_id, quiz_data):
    cursor = conn.cursor()
    try:
        # 1. Insert Quiz
        cursor.execute(
            """
            INSERT INTO quizzes (creator_id, title, description, is_public)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (user_id, quiz_data['title'], quiz_data['description'], False)
        )
        quiz_id = cursor.fetchone()[0]

        # 2. Insert Questions
        for q in quiz_data['questions']:
            cursor.execute(
                """
                INSERT INTO questions (quiz_id, question_text, options, correct_answer, explanation)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (quiz_id, q['question_text'], json.dumps(q['options']), q['correct_answer'], q['explanation'])
            )
        
        conn.commit()
        return quiz_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
