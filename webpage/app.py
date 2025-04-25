from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import json
import asyncio
import httpx
import base64
from uuid import uuid4
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from RAG.rag_system import RAGSystem
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.moonshot.cn/v1",
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

# 初始化Flask应用
app = Flask(__name__)

# 超声知识查询函数
def query_ultrasound_impl(question: str):
    """
    调用DeepSeek医学超声知识API查询专业问题
    """
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

def query_ultrasound_knowledge(arguments: dict[str, any]) -> any:
    try:
        question = arguments["question"]
        result = query_ultrasound_impl(question)
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
    except KeyError:
        return {"error": "参数缺少'question'字段"}
    except Exception as e:
        return {"error": f"内部错误: {str(e)}"}

async def text_to_speech(text):
    """
    使用EdgeTTS将文本转换为语音并返回音频数据
    """
    try:
        import edge_tts
        import io

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
        audio_data = audio_buffer.read()
        
        return {"status": "success", "audio_data": audio_data}
    except Exception as e:
        return {"error": f"语音合成失败: {str(e)}"}

def call_rag_query(question, model_name="BAAI/bge-large-zh-v1.5"):
    try:
        rag = RAGSystem(model_name=model_name)  # 初始化 RAGSystem
        result, picture_path = rag.query(question, k=4)  # k 是返回的相似文本块数量
        return result, picture_path
    except Exception as e:
        print(f"查询失败: {e}")
        return None, []

# 工具映射表
tool_map = {
    "query_ultrasound_knowledge": query_ultrasound_knowledge,
    "text_to_speech": text_to_speech
}

# 配置上传文件夹和文件限制
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # 上传文件保存路径
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大文件大小16MB

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 文件上传端点
@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'files' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    files = request.files.getlist('files')  # 获取文件列表
    uploaded_files = []  # 存储上传文件的元数据
    for file in files:
        if file and file.filename:
            # 确保文件名安全并生成唯一文件名
            original_filename = secure_filename(file.filename)
            extension = os.path.splitext(original_filename)[1]  # 获取文件扩展名
            new_filename = str(uuid4()) + extension  # 生成唯一文件名
            # 保存文件到上传文件夹
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            # 生成文件的访问URL
            file_url = url_for('static', filename='uploads/' + new_filename)
            # 存储原始文件名和URL
            uploaded_files.append({
                'original_name': original_filename,
                'url': file_url
            })
        else:
            return jsonify({'error': '无效文件'}), 400
    # 返回上传成功的文件元数据
    return jsonify({'message': '文件上传成功', 'files': uploaded_files})

# 处理文本和文件的端点
@app.route('/ask', methods=['POST'])
def ask():
    # 获取JSON请求数据
    data = request.get_json()
    text = data.get('text', '')  # 获取文本内容
    files = data.get('files', [])  # 获取文件元数据列表
    page_type = data.get('page_type', 'learning')  # 获取页面类型，默认为learning
    deep_search = data.get('deep_search', False)  # 获取深度搜索标志，默认为False
    enable_tts = data.get('enable_tts', False)  # 获取是否启用语音合成

    # 响应文本
    response_text = ""
    audio_data = None

    # 先判断页面类型
    if page_type == 'learning':
        # 准备消息列表
        messages = [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"}
        ]
        
        # 如果是learning页面，再判断是否需要深度搜索
        messages.append({"role": "user", "content": text})
        
        if deep_search:
            # 深度搜索逻辑：调用RAG系统查询相关知识
            rag_result, picture_paths = call_rag_query(text)
            if rag_result:
                # 将RAG结果添加到消息历史
                messages.append({"role": "system", "content": f"相关知识：\n{rag_result}"})
        
        # 调用AI模型获取回答
        completion = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=messages,
            temperature=0.3,
            tools=tools
        )
        
        # 处理可能的工具调用
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        
        if finish_reason == "tool_calls":
            # 添加AI消息到对话历史
            messages.append(choice.message)
            
            # 处理工具调用
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                if tool_call_name not in tool_map:
                    continue
                    
                # 解析工具调用参数
                tool_call_arguments = json.loads(tool_call.function.arguments)
                tool_function = tool_map[tool_call_name]
                
                # 执行工具调用
                if tool_call_name == "text_to_speech":
                    # 异步执行语音合成
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    tool_result = loop.run_until_complete(tool_function(tool_call_arguments.get("text", "")))
                    loop.close()
                else:
                    # 执行其他工具
                    tool_result = tool_function(tool_call_arguments)
                
                # 添加工具结果到对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result)
                })
            
            # 再次调用AI获取最终回答
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3
            )
        
        # 获取AI回答
        response_text = completion.choices[0].message.content
        
        # 如果是深度搜索且有相关图片，添加图片信息
        if deep_search and picture_paths:
            response_text += "\n\n相关图片：\n" + "\n".join([f"- {path}" for path in picture_paths])
            
        # 如果有文件，添加文件信息
        if files:
            response_text += "\n文件:\n" + "\n".join([f"- {file['original_name']}" for file in files])
    elif page_type == 'usimage':
        # 准备消息列表
        messages = [
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"}
        ]
        
        # usimage页面逻辑：处理图像和文本
        if files and len(files) > 0:
            # 如果有图像文件，处理图像分析
            try:
                # 获取第一个图像文件的URL并转换为本地路径
                image_url = files[0]['url']
                image_path = os.path.join(os.getcwd(), image_url.lstrip('/'))
                
                # 将图像编码为base64
                with open(image_path, "rb") as f:
                    image_data = f.read()
                
                file_ext = os.path.splitext(image_path)[1][1:]  # 获取扩展名（去掉点）
                if not file_ext:
                    file_ext = "png"  # 默认扩展名
                
                image_base64 = f"data:image/{file_ext};base64,{base64.b64encode(image_data).decode('utf-8')}"
                
                # 创建包含图像的消息
                prompt = text if text else "请分析这张超声图像，识别病灶区域和正常区域的特征。"
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                })
                
                # 调用AI模型获取回答（使用支持图像的模型）
                completion = client.chat.completions.create(
                    model="moonshot-v1-128k-vision-preview",
                    messages=messages,
                    temperature=0.3
                )
                
                # 处理可能的工具调用（图像模型通常不支持工具调用，但为了代码一致性添加处理逻辑）
                choice = completion.choices[0]
                finish_reason = choice.finish_reason
                
                if finish_reason == "tool_calls" and hasattr(choice.message, 'tool_calls'):
                    # 添加AI消息到对话历史
                    messages.append(choice.message)
                    
                    # 处理工具调用
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call.function.name
                        if tool_call_name not in tool_map:
                            continue
                            
                        # 解析工具调用参数
                        tool_call_arguments = json.loads(tool_call.function.arguments)
                        tool_function = tool_map[tool_call_name]
                        
                        # 执行工具调用
                        if tool_call_name == "text_to_speech":
                            # 异步执行语音合成
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            tool_result = loop.run_until_complete(tool_function(tool_call_arguments.get("text", "")))
                            loop.close()
                        else:
                            # 执行其他工具
                            tool_result = tool_function(tool_call_arguments)
                        
                        # 添加工具结果到对话历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call_name,
                            "content": json.dumps(tool_result)
                        })
                    
                    # 再次调用AI获取最终回答
                    completion = client.chat.completions.create(
                        model="moonshot-v1-128k-vision-preview",
                        messages=messages,
                        temperature=0.3
                    )
                
                # 获取AI回答
                response_text = completion.choices[0].message.content
            except Exception as e:
                response_text = f"图像分析失败: {str(e)}"
        elif text:
            # 如果只有文本问题，调用超声知识查询
            messages.append({"role": "user", "content": text})
            
            # 调用AI模型获取回答
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3,
                tools=tools
            )
            
            # 获取AI回答
            response_text = completion.choices[0].message.content
        else:
            response_text = "请提供超声图像或相关问题"
    else:
        # 无效页面类型
        response_text = "无效的页面类型"

    # 如果启用了语音合成，生成语音数据
    if enable_tts and response_text:
        try:
            # 使用异步函数需要在事件循环中运行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tts_result = loop.run_until_complete(text_to_speech(response_text))
            loop.close()
            
            if 'audio_data' in tts_result:
                audio_data = tts_result['audio_data']
        except Exception as e:
            print(f"语音合成失败: {str(e)}")

    # 构建响应数据
    response_data = {'text': response_text, 'files': files}
    
    # 如果有音频数据，添加到响应中
    if audio_data:
        # 将二进制音频数据转换为Base64编码
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        response_data['audio'] = audio_base64

    # 返回JSON响应
    return jsonify(response_data)

# 深度搜索端点（兼容旧代码，重定向到/ask）
@app.route('/deepsearch', methods=['POST'])
def deepsearch():
    # 获取JSON请求数据
    data = request.get_json()
    # 添加deep_search标志
    data['deep_search'] = True
    # 调用/ask端点的逻辑
    request.json = data  # 模拟请求数据
    return ask()

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')  # 渲染主页

# 学习页面路由
@app.route('/learning')
def learning():
    return render_template('learning.html')  # 渲染聊天页面

# 影像分析页面路由
@app.route('/usimage')
def usimage():
    return render_template('usimage.html')  # 渲染影像分析页面

# 文本转语音端点
@app.route('/text_to_speech', methods=['POST'])
def generate_speech():
    # 获取JSON请求数据
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': '没有提供文本内容'}), 400
    
    try:
        # 使用异步函数需要在事件循环中运行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tts_result = loop.run_until_complete(text_to_speech(text))
        loop.close()
        
        if 'error' in tts_result:
            return jsonify({'error': tts_result['error']}), 500
            
        if 'audio_data' in tts_result:
            # 将二进制音频数据转换为Base64编码
            import base64
            audio_base64 = base64.b64encode(tts_result['audio_data']).decode('utf-8')
            return jsonify({'audio': audio_base64})
        else:
            return jsonify({'error': '语音合成失败'}), 500
    except Exception as e:
        return jsonify({'error': f'语音合成失败: {str(e)}'}), 500

# 启动Flask应用
if __name__ == '__main__':
    app.run(debug=True)