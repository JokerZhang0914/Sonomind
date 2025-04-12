from typing import *
import json
from openai import OpenAI

client = OpenAI(
    api_key = "sk-t2qDZfR7tnGOJrVMwfiXOwt280seaXTokhR57AFJsyEj9Rh1",
    base_url="https://api.moonshot.cn/v1",
)

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
    }
]

def query_ultrasound_impl(question: str) -> Dict[str, Any]:
    """
    调用DeepSeek医学超声知识API查询专业问题
    """
    import httpx
    
    # DeepSeek API配置
    api_url = "https://api.deepseek.com/v1/medical/ultrasound"
    headers = {
        "Authorization": "sk-27a799972c5a4e0e81a9947d2061954c",  # DeepSeek API密钥
        "Content-Type": "application/json"
    }
    
    payload = {
        "question": question,
        "domain": "ultrasound",
        "language": "zh-CN"
    }
    
    r = httpx.post(api_url, headers=headers, json=payload)
    
    if r.status_code != 200:
        return {"error": f"API请求失败，状态码: {r.status_code}"}
    
    return r.json()

def query_ultrasound_knowledge(arguments: Dict[str, Any]) -> Any:
    question = arguments["question"]
    result = query_ultrasound_impl(question)
    return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}

# 更新tool_map
tool_map = {
    "query_ultrasound_knowledge": query_ultrasound_knowledge
}

messages = [
    {"role": "system",
     "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"}
]

# 从终端获取用户输入
while True:
    user_input = input("\n请输入您的问题(输入'退出'结束): ")
    if user_input.lower() == '退出':
        break
        
    messages.append({"role": "user", "content": user_input})
    
    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        completion = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=messages,
            temperature=0.3,
            tools=tools,
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        print(f"模型返回的finish_reason: {finish_reason}")
        
        if finish_reason == "tool_calls":
            print("检测到工具调用请求")
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                print(f"正在调用工具: {tool_call_name}")
                tool_call_arguments = json.loads(tool_call.function.arguments)
                tool_function = tool_map[tool_call_name]
                tool_result = tool_function(tool_call_arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),
                })

    print("\nKimi回答:", choice.message.content)
    print("="*50)