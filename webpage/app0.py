from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import json
import asyncio
import base64
from uuid import uuid4
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 导入自定义服务模块
from src.chat.chat_service import client, tools, tool_map as chat_tool_map
from src.image.image_service import encode_image_to_base64, has_image_content
from src.RAG.rag_system import call_rag_query

# 初始化Flask应用
app = Flask(__name__)

# 配置上传文件夹和文件限制
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static/uploads')  # 上传文件保存路径
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

# 更新工具映射表
tool_map = chat_tool_map.copy()

# 处理文本和文件的端点
@app.route('/ask', methods=['POST'])
def ask():
    # 获取JSON请求数据
    data = request.get_json()
    text = data.get('text', '')  # 获取文本内容
    files = data.get('files', [])  # 获取文件元数据列表
    page_type = data.get('page_type', 'learning')  # 获取页面类型，默认为learning
    deep_search = data.get('deep_search', False)  # 获取深度搜索标志，默认为False

    # 响应文本
    response_text = ""

    # 先判断页面类型
    if page_type == 'learning':
        # 准备消息列表
        messages = [
            {"role": "system", "content": "你是一个医学超声领域的AI助手，擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"}
        ]
        
        # 处理图像和文本
        if files and len(files) > 0:
            # 如果有图像文件，处理图像分析
            try:
                # 获取第一个图像文件的URL并提取文件名
                image_url = files[0]['url']
                # 从URL中提取文件名
                filename = image_url.split('/')[-1]
                # 使用配置的上传文件夹路径构建绝对路径
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # 使用image_service将图像编码为base64
                image_base64 = encode_image_to_base64(image_path)
                
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
                
                # 获取AI回答
                response_text = completion.choices[0].message.content
            except Exception as e:
                response_text = f"图像分析失败: {str(e)}"
        else:
            # 如果没有图像，只处理文本问题
            messages.append({"role": "user", "content": text})
            
            # 如果是learning页面，再判断是否需要深度搜索
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
            {"role": "system", "content": "你是一个医学超声领域的AI助手，擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"}
        ]
        
        # usimage页面逻辑：处理图像和文本
        if files and len(files) > 0:
            # 如果有图像文件，处理图像分析
            try:
                # 获取第一个图像文件的URL并提取文件名
                image_url = files[0]['url']
                # 从URL中提取文件名
                filename = image_url.split('/')[-1]
                # 使用配置的上传文件夹路径构建绝对路径
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # 使用image_service将图像编码为base64
                image_base64 = encode_image_to_base64(image_path)
                
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

    # 返回Markdown格式的文本和文件元数据
    return jsonify({'text': response_text, 'files': files})

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

# 启动Flask应用
if __name__ == '__main__':
    app.run(debug=True)