import sys
import os

# รันสคริปต์นี้จากโฟลเดอร์ Mobile_server
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from app.service.chaobot import ask_chatbot
from app.service.evaluation import evaluate_batch
import time

def add_space_to_thai(text: str) -> str:
    """ 
    Trick สำหรับภาษาไทย: 
    ในภาษาไทยคำติดกัน ROUGE จะมองเป็นคำเดียว ให้เราลองตัดคำด้วยช่องว่างก่อน
    ถ้ามี pythainlp จะดีมาก แต่ถ้าไม่มี ให้ประเมินระดับตัวอักษรแทน (Character-level)
    """
    try:
        from pythainlp.tokenize import word_tokenize
        return " ".join(word_tokenize(text))
    except ImportError:
        # ถ้าไม่ได้ลง pythainlp ถือว่าแยกตัวอักษรแบบหยาบๆ ตัวต่อตัว
        return " ".join(list(text))

# ==========================================
# ใส่ข้อมูลทดสอบ (Test Set) สำหรับพรีเซนต์
# ==========================================
test_cases = [
    {
        "question": "แอปพลิเคชันนี้ใช้งานอย่างไร?",
        "reference_answer": """แอปพลิเคชันนี้คือ "Learnify Bot" — ผู้ช่วยส่วนตัวที่ออกแบบมาเพื่อช่วยคุณเรียนรู้และเข้าใจเนื้อหาต่าง ๆ ได้ง่ายขึ้น!  

คุณสามารถใช้งานได้โดย:  
- ถามคำถามเกี่ยวกับบทเรียน วิชาการ หรือแนวข้อสอบ  
- อัปโหลดเอกสาร (PDF/Text) เพื่อให้ฉันช่วยวิเคราะห์ สรุป หรือตอบคำถามจากเนื้อหานั้น  
- ขอคำอธิบายเพิ่มเติม ตัวอย่าง หรือสรุปสั้น ๆ ตามต้องการ  

แค่พิมพ์สิ่งที่อยากเรียนรู้ ฉันจะช่วยคุณทันที 😊  
หากมีเอกสารอะไรมา บอกได้เลย — ฉันพร้อมช่วยวิเคราะห์ทันที!"""
    },
    {
        "question": "มีระบบอะไรบ้าง?",
        "reference_answer": """ไม่พบข้อมูลนี้ในเอกสารที่อัปโหลด  
หากคุณมีเอกสารหรือรายละเอียดเพิ่มเติม (เช่น แผนผังระบบ หรือคำอธิบายวิชา) สามารถอัปโหลดมาได้เลยนะครับ ฉันจะช่วยวิเคราะห์และอธิบายให้เข้าใจง่ายขึ้นทันที! 📎✨"""
    }
]

def run_test():
    session_id = "eval_test@learnify.com"
    pairs_for_batch = []
    
    print("-" * 50)
    print("🚀 เริ่มทำการประเมิน RAG Chatbot...")
    print("-" * 50)
    
    for idx, tc in enumerate(test_cases, 1):
        q = tc["question"]
        ref = tc["reference_answer"]
        print(f"[{idx}/{len(test_cases)}] ถาม: {q}")
        
        # 1. ให้ AI ตอบ
        start_time = time.time()
        try:
            bot_response = ask_chatbot(session_id=session_id, question=q)
            # ดึงเฉพาะคำความหมายจาก dictionary ที่ AI ตอบกลับมา
            rag_answer = bot_response.get("answer", str(bot_response))
        except Exception as e:
            rag_answer = f"Error: {e}"
        time_taken = time.time() - start_time
        
        # 2. เตรียมข้อความภาษาไทยให้พร้อมกับการประเมินโดยเว้นวรรค
        spaced_ref = add_space_to_thai(ref)
        spaced_rag = add_space_to_thai(rag_answer)
        
        print(f"💡 คำตอบ AI: {rag_answer}")
        print(f"⏱️ ใช้เวลา: {time_taken:.2f} วินาที")
        print("-" * 20)
        
        pairs_for_batch.append({
            "reference": spaced_ref,
            "hypothesis": spaced_rag
        })

    # 3. คำนวณ ROUGE Score ทีเดียวจบ
    result = evaluate_batch(pairs_for_batch)
    
    # 4. แสดงผลลัพธ์
    print("\n✅ สรุปผลลัพธ์ Evaluation (ROUGE-L เฉลี่ย)\n" + "=" * 50)
    
    rougeL_avg = result["average_scores"].get("rougeL", {})
    precision = rougeL_avg.get('precision', 0)
    recall = rougeL_avg.get('recall', 0)
    f1 = rougeL_avg.get('fmeasure', 0)
    
    print(f"🎯 Precision (ความแม่นยำ - AI ไม่ตอบเกินจริง): {precision:.4f} ({precision*100:.1f}%)")
    print(f"🎯 Recall    (ความครอบคลุม - AI ตอบครบประเด็น): {recall:.4f} ({recall*100:.1f}%)")
    print(f"🏆 F1-Score  (ภาพรวมประสิทธิภาพ): {f1:.4f} ({f1*100:.1f}%)")
    print("=" * 50)
    print("คำแนะนำสำหรับการพรีเซนต์: นำเปอร์เซ็นต์เหล่านี้ไปใส่สไลด์แผนภูมิแท่ง (Bar Chart) เพื่อโชว์ความแม่นยำของ AI ครับ")

if __name__ == "__main__":
    run_test()
