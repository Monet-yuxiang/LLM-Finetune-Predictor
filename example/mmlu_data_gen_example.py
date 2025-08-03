import os
import json
import time
import random
from typing import List, Dict
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from datetime import datetime

class DialogueGenerator:
    def __init__(self):
        load_dotenv()
        self.api_url = os.getenv('API_URL')
        self.api_key = os.getenv('API_KEY')
        self.llm_model = "DeepSeek-V3"
        self.retry_limit = 3

        # subject 类别（参考 MMLU）

        self.subjects = [
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history"
        ]

        self._validate_env_vars()

    def _validate_env_vars(self):
        required_vars = ['API_URL', 'API_KEY']
        missing = [v for v in required_vars if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    def _get_subject(self) -> str:
        return random.choice(self.subjects)

    def _get_system_prompt(self) -> str:
        return """You are a professional multiple-choice question generator.

Generate a multiple-choice question in the following JSON format:

{
  "question": "Which of the following best describes...",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "C"
}

Requirements:
1. The question must be related to the given subject.
2. Include exactly 4 plausible, distinct choices.
3. Answer must be one of A/B/C/D.
4. Do not include explanations or any other text.
5. Output valid JSON only (no markdown formatting)."""

    def _fix_json_format(self, content: str, subject: str) -> str:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            parsed = json.loads(content)
            if self._validate_qa_pair(parsed):
                parsed["subject"] = subject
                return json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            print("Format error:", e)
            raise ValueError("Invalid JSON format")
        raise ValueError("Unable to parse valid QA object")

    def _validate_qa_pair(self, qa: Dict) -> bool:
        if not isinstance(qa, dict):
            return False
        if "question" not in qa or "choices" not in qa or "answer" not in qa:
            return False
        if not isinstance(qa["choices"], list) or len(qa["choices"]) != 4:
            return False
        if qa["answer"] not in ["A", "B", "C", "D"]:
            return False
        return True

    def _generate_conversation(self) -> Dict:
        subject = self._get_subject()

        # 添加控制维度
        reasoning_type = random.choice([
            "factual recall", "causal reasoning", "definition-based",
            "application-based", "comparative reasoning", "mechanism explanation"
        ])

        question_focus = random.choice([
            "key theory", "formula or equation", "real-world example",
            "historical context", "practical implication", "scientific method"
        ])
  
        question_templates = random.choice([
    "Which of the following best describes...",
    "What is the primary cause of...",
    "Which statement is true about...",
    "How does...",
    "Why does...",
    "In the context of...",
    "Given the scenario, which option correctly explains...",
    "What can be inferred from the...",
    "Which example illustrates the concept of...",
    "Identify the correct explanation for...",
])



        prompt = f"""Generate one multiple-choice question for the subject: {subject}
        Requirements:
- Reasoning type: {reasoning_type}
- Focus of the question: {question_focus}
- Question template: {question_templates}
- The question should be original and distinct from typical textbook examples
- Follow this format strictly:

{{
  "question": "...",
  "choices": ["...", "...", "...", "..."],
  "answer": "B"
}}

Do not provide any explanation or text outside of this format."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        data = {
            "model": self.llm_model,
            "messages": messages
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content.strip():
            raise ValueError("Empty content returned from model")

        return json.loads(self._fix_json_format(content, subject))

    def _generate_with_retry(self) -> Dict:
        for attempt in range(self.retry_limit):
            try:
                qa = self._generate_conversation()
                if self._validate_qa_pair(qa):
                    return qa
            except Exception as e:
                print(f"Retry {attempt + 1}/{self.retry_limit} failed: {e}")
                time.sleep(1)
        raise ValueError("Failed to generate QA pair after retries")

    def generate_dataset(self, num_examples: int, dataset_name: str = None):
        if dataset_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"mmlu_{timestamp}"

        output_path = "datasets/experiment/mmlu"
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"{dataset_name}.jsonl")

        dataset_chunk = []
        count = 0
        pbar = tqdm(total=num_examples, desc="Generating MMLU-format QA")

        while count < num_examples:
            try:
                qa = self._generate_with_retry()
                dataset_chunk.append(qa)
                count += 1
                pbar.update(1)

                # 每50条写入一次
                if len(dataset_chunk) >= 50 or count == num_examples:
                    self.append_dataset_chunk(dataset_chunk, save_path)
                    dataset_chunk = []

                time.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)
                continue

        pbar.close()
        print(f"✅ Final dataset saved to: {save_path}")

    def append_dataset_chunk(self, chunk: List[Dict], path: str):
        with open(path, 'a', encoding='utf-8') as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(chunk)} entries to {path}")


    def save_dataset(self, dataset: List[Dict], dataset_name: str = None):
        if dataset_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"mmlu_{timestamp}"

        output_path = "datasets/experiment/mmlu"
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"{dataset_name}.jsonl")

        with open(path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Dataset saved to: {path}")


def main():
    try:
        generator = DialogueGenerator()
        num_examples = 500  # ✅ 可修改为 100, 1000 等

        print(f"Generating {num_examples} MMLU-style multiple-choice QA examples...")
        generator.generate_dataset(num_examples=num_examples)

    except Exception as e:
        print(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
