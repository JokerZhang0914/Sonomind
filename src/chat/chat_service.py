from typing import *
from openai import OpenAI
import json
import httpx
from config import API_KEY, DEEPSEEK_API_KEY, MOONSHOT_BASE_URL, DEEPSEEK_API_URL

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=MOONSHOT_BASE_URL,
)

# 定义工具列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "query_ultrasound_knowledge",
            "description": """ 
                查询医学超声相关专业知识。
                当用户询问关于医学超声、超声检查、超声诊断等技术问题时，调用此工具。
                工具会连接DeepSeek医学知识库，获取权威、准确的医学超声知识回答。
            """,
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户提出的关于医学超声的具体问题，需要准确描述问题内容"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_speech",
            "description": "将文本转换为语音并播放",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "需要转换为语音的文本内容"
                    }
                }
            }
        }
    }
]

def query_ultrasound_impl(question: str) -> Dict[str, Any]:
    """调用DeepSeek医学超声知识API查询专业问题"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "question": question,
        "domain": "ultrasound",
        "language": "zh-CN"
    }
    
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP错误: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"请求失败: {str(e)}"}

def query_ultrasound_knowledge(arguments: Dict[str, Any]) -> Any:
    try:
        question = arguments["question"]
        result = query_ultrasound_impl(question)
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
    except KeyError:
        return {"error": "参数缺少'question'字段"}
    except Exception as e:
        return {"error": f"内部错误: {str(e)}"}

# 工具映射表
tool_map = {
    "query_ultrasound_knowledge": query_ultrasound_knowledge,
}