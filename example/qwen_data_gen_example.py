"""
1. Load environment variables
2. Generate context-based QA data using API URL and API key from .env
3. Save the generated dataset
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
import requests
from tqdm import tqdm
import re
from datetime import datetime
from collections import defaultdict

class DialogueGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.api_url = os.getenv('API_URL') 
        self.api_key = os.getenv('API_KEY') 
        self.llm_model = "DeepSeek-V3" # gpt-4o-mini
        
        # 用于追踪每个话题生成的context
        self.topic_contexts = defaultdict(set)
        
        # Validate required environment variables
        self._validate_env_vars()

    def _validate_env_vars(self):
        """Validate required environment variables"""
        required_vars = ['API_URL', 'API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _fix_json_format(self, content: str) -> str:
        """修复JSON格式问题：
        1. 移除可能的markdown代码块标记
        2. 保持JSON本身的格式完整性
        """
        # 移除markdown代码块标记
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        return content.strip()

    def _get_example_format(self) -> str:
        """返回标准的QA对示例格式"""
        return """{
    "messages": [
        {
            "role": "user",
            "content": "Context: The Great Wall of China was built over many centuries by different Chinese dynasties. The most famous sections were built during the Ming Dynasty (1368-1644). The wall served multiple purposes: it protected against invasions, regulated trade along the Silk Road, and controlled immigration and emigration. The total length of the wall built during the Ming Dynasty is around 8,850 kilometers.\\nQuestion: How long is the Great Wall section built during the Ming Dynasty?"
        },
        {
            "role": "assistant",
            "content": "8,850 kilometers."
        }
    ]
}"""

# 1. Content Diversity:
#    - Time Periods: Ancient, Medieval, Modern, Contemporary
#    - Regions: Asia, Europe, Americas, Africa, Oceania
#    - Perspectives: Politics, Economy, Culture, Technology, Society
#    - Scales: Major Events, Daily Life, Individual Stories

# 2. Question Types:
#    - Factual: dates, numbers, statistics, specific details
#    - Analytical: causes, effects, influences, relationships
#    - Comparative: differences, changes, developments
#    - Evaluative: significance, impact, importance
#    - Procedural: processes, methods, techniques

# 3. Format Requirements:
#    - Context: 50-100 words, specific and engaging
#    - Question: clear and focused
#    - Answer: concise and short (words extracted from context)

    def _get_system_prompt(self) -> str:
        """返回系统prompt"""
        return """You are a specialized encyclopedia-type QA pair generator. Generate diverse QA pairs with the following requirements:

1. Content Diversity: you will first generate a context, follow by a question and an answer.
2. Format Requirements:
   - Context: 100-150 words, specific and engaging
   - Question: clear and focused
   - Answer: concise and short (words extracted from context)

Example format:
{
    "messages": [
        {
            "role": "user",
            "content": "Context: [specific historical context]\\nQuestion: [focused question]"
        },
        {
            "role": "assistant",
            "content": "[concise answer]"
        }
    ]
}"""

    def _generate_conversation(self, topic_type: str, time_period: str = None, region: str = None) -> Dict:
        """Generate a context-based QA pair
        
        Args:
            topic_type: 具体的主题
            time_period: 可选的时间段
            region: 可选的地区
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Define diverse perspectives and approaches
        perspectives = {
            "time_periods": [
                "Ancient Times (pre-500 CE)", "Middle Ages (500-1500)",
                "Early Modern (1500-1800)", "Modern Era (1800-1945)",
                "Contemporary Period (1945-present)"
            ],
            "regions": [
                "East Asia", "South Asia", "Middle East", "Europe",
                "North America", "South America", "Africa", "Oceania"
            ],
            "aspects": [
                "Political Systems", "Economic Development",
                "Cultural Exchange", "Technological Innovation",
                "Social Movements", "Daily Life", "Environmental Change",
                "Scientific Progress", "Artistic Achievement"
            ],
            "scales": [
                "Major Events", "Daily Life", "Individual Stories"
            ],
            "themes": [
                "Innovation and Progress", "Conflict and Resolution",
                "Tradition and Change", "Power and Authority",
                "Identity and Culture", "Cause and Effect",
                "Adaptation and Survival", "Exchange and Trade"
            ],
            "perspectives": [
                "Social Impact", "Economic Influence",
                "Cultural Significance", "Political Implications",
                "Environmental Effects", "Technological Advancement",
                "Historical Legacy", "Global Connections"
            ],
            "focus_areas": [
                "People and Society", "Art and Literature",
                "Science and Technology", "War and Peace",
                "Religion and Philosophy", "Education and Learning",
                "Commerce and Trade", "Environment and Resources"
            ],
            "narrative_angles": [
                "Historical Development", "Comparative Analysis",
                "Case Study", "Biographical Account",
                "Cultural Perspective", "Scientific Analysis",
                "Social Commentary", "Economic Analysis"
            ]
        }
        
        # 随机选择多个维度来构建更丰富的prompt
        # 从每个维度组中随机选择一个元素，但确保不会选到重复的概念
        selected_elements = []
        
        # 基础维度：时间和地区
        if time_period is None:
            time_period = random.choice(perspectives["time_periods"])
        if region is None:
            region = random.choice(perspectives["regions"])
            
        # 从其他维度中随机选择两个不同的组合
        available_dimensions = [
            ("aspects", "aspects", "主要方面"),
            ("themes", "themes", "主题"),
            ("perspectives", "perspectives", "视角"),
            ("focus_areas", "focus_areas", "关注点"),
            ("narrative_angles", "narrative_angles", "分析角度"),
            ("scales", "scales", "规模")
        ]
        
        # 随机选择两个不同的维度
        selected_dimensions = random.sample(available_dimensions, 3)
        
        # 从选中的维度中各选择一个元素
        primary_dimension = random.choice(perspectives[selected_dimensions[0][0]])
        secondary_dimension = random.choice(perspectives[selected_dimensions[1][0]])
        
        # Question patterns for different types of inquiry
        question_patterns = {
            "factual": [
                "What was the specific [number/date/statistic] of [event/phenomenon]?",
                "When did [event/development] occur?",
                "Who were the key figures in [event/period]?"
            ],
            "analytical": [
                "How did [factor] influence [outcome]?",
                "What were the main causes of [event/change]?",
                "Why did [event/development] occur in this way?"
            ],
            "comparative": [
                "How did [aspect] change from [period1] to [period2]?",
                "What were the differences between [element1] and [element2]?",
                "How did [factor] vary across [regions/times]?"
            ],
            "evaluative": [
                "What was the significance of [event/development]?",
                "How did [event/change] impact [society/field]?",
                "Why was [element] important for [context]?"
            ]
        }
        
        # 选择问题类型和模板
        question_type = random.choice(list(question_patterns.keys()))
        question_template = random.choice(question_patterns[question_type])
        
        # 构建动态提示词
        focus_prompt = f"Focus on {region} during {time_period}, analyzing {primary_dimension} from {secondary_dimension} perspective"
        question_prompt = f"Use this pattern: {question_template}"
        
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": f"Generate a QA pair about {topic_type}. {focus_prompt}.{question_prompt}. Requirements: 1) Specific and engaging context 2) Clear and focused question 3) Concise answer"
            }
        ]
        
        data = {
            "model": self.llm_model,
            "messages": messages
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content.strip():
                raise ValueError("API returned empty content")
            
            content = self._fix_json_format(content)
            
            try:
                parsed_result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                print(f"处理后的内容: {content}")
                raise
            
            if not self._validate_qa_pair(parsed_result):
                raise ValueError("Invalid QA pair format")
                
            context = self._extract_context(parsed_result)
            if context in self.topic_contexts[topic_type]:
                raise ValueError("Duplicate context detected")
                
            self.topic_contexts[topic_type].add(context)
            
            return parsed_result
                
        except Exception as e:
            print(f"Failed to generate QA pair: {str(e)}")
            raise

    def _extract_context(self, qa_pair: Dict) -> str:
        """从QA对中提取context部分"""
        try:
            content = qa_pair["messages"][0]["content"]
            context = content.split("Question:")[0].replace("Context:", "").strip()
            return context
        except Exception:
            return ""

    def generate_dataset(self, num_examples: int, topic_type: str = None, time_period: str = None, region: str = None) -> pd.DataFrame:
        """Generate the complete dataset
        
        Args:
            num_examples: 要生成的QA对数量
            topic_type: 可选的指定主题
            time_period: 可选的指定时间段
            region: 可选的指定地区
        """
        dataset_rows = []
        successful_generations = 0
        
        pbar = tqdm(total=num_examples, desc="Generating QA pairs")
        while successful_generations < num_examples:
            try:
                # Generate QA pair
                qa_pair = self._generate_conversation(
                    topic_type=topic_type,
                    time_period=time_period,
                    region=region
                )
                
                # 验证QA pair格式
                if not self._validate_qa_pair(qa_pair):
                    print(f"\nSkipping invalid QA pair format")
                    continue
                
                # Add to dataset
                dataset_rows.append(qa_pair)
                successful_generations += 1
                pbar.update(1)
                print(f"\nSuccessfully generated QA pair {successful_generations}/{num_examples}")
                
                time.sleep(random.randint(1, 3))  # Avoid API rate limits
                
            except Exception as e:
                print(f"\nError generating QA pair: {str(e)}")
                time.sleep(2)  # 固定等待时间
                continue
        
        pbar.close()
        
        if not dataset_rows:
            raise ValueError("Failed to generate any valid QA pairs")
        
        return pd.DataFrame(dataset_rows)

    def _validate_qa_pair(self, qa_pair: Dict) -> bool:
        """验证QA pair的格式是否正确"""
        try:
            messages = qa_pair.get("messages", [])
            if len(messages) != 2:
                return False
            
            user_msg = messages[0]
            assistant_msg = messages[1]
            
            if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
                return False
            
            user_content = user_msg.get("content", "")
            if not user_content.startswith("Context:") or "Question:" not in user_content:
                return False
            
            if not assistant_msg.get("content", "").strip():
                return False
            
            return True
        except Exception:
            return False

    def save_dataset(self, df: pd.DataFrame, dataset_name: str = None):
        """Save the dataset locally"""
        if dataset_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"qa_dataset_{timestamp}"
        
        # Save locally
        local_path = f"datasets/{self.llm_model}"
        # local_path = f"datasets"
        os.makedirs(local_path, exist_ok=True)
        
        # Save as JSON
        json_path = f"{local_path}/{dataset_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict('records'), f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to: {json_path}")

def main():
    try:
        # Initialize generator
        generator = DialogueGenerator()
        
        # 示例：生成特定主题和时期的QA对
        print("示例1：生成特定主题和时期的QA对")
        df = generator.generate_dataset(
            num_examples=100,
        )

        
        if len(df) > 0:
            # Display sample
            print("\nSample QA pair:")
            print(json.dumps(df.iloc[0].to_dict(), ensure_ascii=False, indent=2))
            
            # Save locally
            generator.save_dataset(df)
        else:
            print("No valid QA pairs were generated.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

