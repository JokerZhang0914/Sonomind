from typing import *
import json
import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

client = OpenAI(
    api_key = os.getenv("API_KEY"),
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
    """
    调用DeepSeek医学超声知识API查询专业问题
    """
    import httpx
    
    # DeepSeek API配置
    api_url = "https://api.deepseek.com/v1/medical/ultrasound"
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "question": question,
        "domain": "ultrasound",
        "language": "zh-CN"
    }
    
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(api_url, headers=headers, json=payload)
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

async def text_to_speech(text: str) -> Dict[str, Any]:
    """使用EdgeTTS将文本转换为语音并直接播放"""
    try:
        import edge_tts
        import asyncio
        import pygame
        import io
        
        # 初始化pygame音频系统
        pygame.mixer.init()
        
        # 使用中文女声
        voice = "zh-CN-XiaoxiaoNeural"
        communicate = edge_tts.Communicate(text, voice)
        
        # 创建内存缓冲区存储音频数据
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        
        # 将缓冲区指针移到开始位置
        audio_buffer.seek(0)
        
        # 使用pygame播放音频
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)
            
        return {"status": "success", "message": "语音播放成功"}
    except Exception as e:
        return {"error": f"语音合成失败: {str(e)}"}

# 更新tool_map
tool_map = {
    "query_ultrasound_knowledge": query_ultrasound_knowledge,
    "text_to_speech": text_to_speech
}

messages = [
    {"role": "system",
     "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"}
]

# 从终端获取用户输入
while True:
    user_input = input("\n请输入您的问题(输入'退出'('q')结束): ")
    if user_input.lower() in ['退出', 'q', 'Q']:
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
                if tool_call_name not in tool_map:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "错误：未知工具",
                    })
                    continue
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
    
    # 使用EdgeTTS朗读回答
    asyncio.run(text_to_speech(choice.message.content))
