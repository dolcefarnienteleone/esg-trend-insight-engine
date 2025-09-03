import os
import json
import spacy
from anthropic import Anthropic
from dotenv import load_dotenv
import ast
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('triplets_extraction.log'),
        logging.StreamHandler()
    ]
)

# Setup
load_dotenv()
nlp = spacy.load("en_core_web_sm")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

input_folder = "./processed_text_docs"
output_folder = "./triples_output"
os.makedirs(output_folder, exist_ok=True)

# 緩存已處理的句子，避免重複處理
processed_cache = {}

def generate_prompt(text_chunk: str, named_entities: List[str]) -> str:
    """生成提示詞，使用更精確的指導"""
    prompt = f"""
You are an AI assistant that extracts structured knowledge from ESG reports.

Given the following text, identify subject–predicate–object triples that represent ESG-related facts. Use only the context provided. Focus on relationships like goals, commitments, suppliers, metrics, timelines, etc.

Text:
{text_chunk}

Named Entities: {', '.join(named_entities)}

IMPORTANT: Return ONLY a Python list of tuples in this exact format:
[("Subject", "Predicate", "Object"), ("Subject2", "Predicate2", "Object2")]

Rules:
1. Only extract meaningful ESG-related relationships
2. Keep subjects and objects concise
3. Use present tense for predicates
4. Do not include any explanatory text
5. Ensure all triples are complete and meaningful
"""
    return prompt.strip()

def safe_eval_triples(result: str) -> List[Tuple[str, str, str]]:
    """安全地解析返回的三元組"""
    try:
        # 清理結果字符串
        result = result.strip()
        if not result.startswith('['):
            return []
            
        # 首先嘗試使用 ast.literal_eval 解析
        extracted = ast.literal_eval(result)
    except:
        try:
            # 如果失敗，嘗試使用 json.loads
            extracted = json.loads(result)
        except:
            return []
    
    # 確保結果是列表格式
    if not isinstance(extracted, list):
        return []
    
    # 驗證每個三元組的格式
    valid_triples = []
    for triple in extracted:
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            # 確保所有元素都是字符串
            if all(isinstance(x, str) for x in triple):
                valid_triples.append(tuple(triple))
    
    return valid_triples

def process_sentence(sent_text: str, entities: List[str], max_retries: int = 3) -> List[Dict]:
    """處理單個句子"""
    # 檢查緩存
    cache_key = f"{sent_text}_{','.join(sorted(entities))}"
    if cache_key in processed_cache:
        return processed_cache[cache_key]

    triples = []
    prompt = generate_prompt(sent_text, entities)
    
    for attempt in range(max_retries):
        try:
            # 添加延遲以避免速率限制
            if attempt > 0:
                time.sleep(3)
                
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=300,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.content[0].text.strip()
            
            extracted = safe_eval_triples(result)
            
            for triple in extracted:
                triples.append({
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                    "source_sentence": sent_text
                })
            
            # 如果成功提取到三元組，存入緩存
            if triples:
                processed_cache[cache_key] = triples
            
            break
            
        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(f"Error processing sentence after {max_retries} attempts: {sent_text[:100]}... → {e}")
            time.sleep(2)  # 增加延遲以避免速率限制
            continue
    
    return triples

def extract_triples(text: str, filename: str) -> Tuple[int, str]:
    """提取文本中的三元組"""
    doc = nlp(text)
    sentences = list(doc.sents)
    all_triples = []
    
    # 使用線程池並行處理句子，減少並行數以避免速率限制
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for sent in sentences:
            sent_text = sent.text.strip()
            entities = [ent.text for ent in sent.ents]
            
            if len(entities) >= 2:
                futures.append(
                    executor.submit(process_sentence, sent_text, entities)
                )
        
        # 收集結果
        for future in futures:
            triples = future.result()
            all_triples.extend(triples)

    # 保存結果
    output_path = os.path.join(output_folder, filename.replace(".txt", "_triples.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)

    return len(all_triples), output_path

def main():
    """主函數"""
    start_time = time.time()
    total_triples = 0
    
    for i, file in enumerate(os.listdir(input_folder)):
        if file.endswith(".txt"):  # 處理所有txt文件
            logging.info(f"Processing file: {file}")
            with open(os.path.join(input_folder, file), "r", encoding="utf-8") as f:
                raw_text = f.read()
                count, out_path = extract_triples(raw_text, file)
                total_triples += count
                logging.info(f"{file} → {count} triples extracted → saved to {out_path}")
    
    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    logging.info(f"Total triples extracted: {total_triples}")

if __name__ == "__main__":
    main() 