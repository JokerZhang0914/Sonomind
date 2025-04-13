from typing import *
import json
import asyncio
from src.chat.chat_service import client, tools, tool_map
from src.speech.speech_service import text_to_speech

# 系统提示信息
system_message = {
    "role": "system",
    "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"
}

# 初始化消息列表
messages = [system_message]

# 更新工具映射表，添加语音功能
tool_map["text_to_speech"] = text_to_speech

async def main():
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
                    
                    # 处理异步函数调用
                    if asyncio.iscoroutinefunction(tool_function):
                        tool_result = await tool_function(**tool_call_arguments)
                    else:
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
        await text_to_speech(choice.message.content)

if __name__ == "__main__":
    asyncio.run(main())