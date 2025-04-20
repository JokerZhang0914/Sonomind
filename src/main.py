from typing import *
import json
import asyncio
from chat.chat_service import client, tools, tool_map
from speech.speech_service import text_to_speech
from image import encode_image_to_base64, has_image_content

# 系统提示信息
system_message = {
    "role": "system",
    "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"
}

# 初始化消息列表
messages = [system_message]

# 更新工具映射表，添加语音功能
tool_map["text_to_speech"] = text_to_speech

async def main():
    while True:
        print("\n请选择操作:")
        print("1. 输入文本问题")
        print("2. 上传超声图像进行分析")
        print("3. 退出")
        
        choice = input("请输入选项(1/2/3): ")
        
        if choice == "3":
            break
        
        if choice == "1":
            user_input = input("请输入您的问题: ")
            messages.append({"role": "user", "content": user_input})
        
        elif choice == "2":
            image_path = input("请输入图像文件路径: ")
            try:
                image_url = encode_image_to_base64(image_path)
                prompt = input("请输入关于图像的问题(默认为'请分析这张超声图像'): ") or "请分析这张超声图像，识别病灶区域和正常区域的特征。"
                
                # 创建包含图像的消息
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                })
                print(f"图像已上传: {image_path}")
            except Exception as e:
                print(f"图像上传失败: {str(e)}")
                continue
        else:
            print("无效选项，请重新选择")
            continue
        
        finish_reason = None
        while finish_reason is None or finish_reason == "tool_calls":
            # 根据是否有图像选择不同的模型
            model = "moonshot-v1-128k-vision-preview" if has_image_content(messages) else "moonshot-v1-128k"
            
            # 创建请求参数
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
            }
            
            # 只在非图像模式下添加工具
            if not has_image_content(messages):
                request_params["tools"] = tools
                
            completion = client.chat.completions.create(**request_params)
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