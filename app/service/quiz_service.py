import os
import json
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

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
     "คุณคืออาจารย์ผู้เชี่ยวชาญด้านการออกข้อสอบ หน้าที่ของคุณคืออ่านเนื้อหาที่ได้รับแล้วสร้างข้อสอบแบบปรนัย (4 ตัวเลือก) จำนวน 5-10 ข้อ\n"
     "ข้อกำหนด:\n"
     "1. สร้างคำถามที่วัดความเข้าใจ ไม่ใช่แค่การจำ\n"
     "2. ให้คำอธิบาย (explanation) ที่ชัดเจนสำหรับแต่ละข้อ\n"
     "3. ผลลัพธ์ต้องเป็นรูปแบบ JSON ตามที่กำหนดเท่านั้น\n"
     "4. ใช้ภาษาไทยในการออกข้อสอบ"),
    ("human", "เนื้อหาสำหรับออกข้อสอบ:\n{context}\n\n{format_instructions}")
])

quiz_chain = prompt | llm | parser

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_quiz_from_text(text: str) -> dict:
    # Limit text to avoid token overflow (approx 4000 chars for safety)
    truncated_text = text[:8000] 
    
    response = quiz_chain.invoke({
        "context": truncated_text,
        "format_instructions": parser.get_format_instructions()
    })
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
            (user_id, quiz_data['title'], quiz_data['description'], True)
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
