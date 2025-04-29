from flask import jsonify, url_for
from werkzeug.utils import secure_filename
import os
import json
from uuid import uuid4
from typing import Optional, Tuple, List, Dict, Any

from src.chat.chat_service import client, tools, tool_map as chat_tool_map
from src.image.image_service import encode_image_to_base64
from src.RAG.rag_system import call_rag_query, copy_images_to_static
# from src.RAG.image_utils import copy_images_to_static

class PageHandler:
    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        self.tool_map = chat_tool_map.copy()
        self.system_message = {
            "role": "system",
            "content": "你是一个医学超声领域的AI助手，擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。你具备医学超声图像分析能力，可以分析已分割好病灶和正常区域的超声图像。"
        }

    def handle_file_upload(self, files) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """处理文件上传"""
        uploaded_files = []
        for file in files:
            if file and file.filename:
                original_filename = secure_filename(file.filename)
                extension = os.path.splitext(original_filename)[1]
                new_filename = str(uuid4()) + extension
                file_path = os.path.join(self.upload_folder, new_filename)
                file.save(file_path)
                file_url = url_for('static', filename=f'uploads/{new_filename}')
                uploaded_files.append({
                    'original_name': original_filename,
                    'url': file_url
                })
            else:
                return [], "无效文件"
        return uploaded_files, None

    def process_image(self, image_url: str, text: str) -> str:
        """处理图像分析"""
        try:
            filename = image_url.split('/')[-1]
            image_path = os.path.join(self.upload_folder, filename)
            image_base64 = encode_image_to_base64(image_path)
            
            messages = [self.system_message]
            prompt = text if text else "请分析这张超声图像，识别病灶区域和正常区域的特征。"
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_base64}
                    },
                    {"type": "text", "text": prompt}
                ]
            })
            
            completion = client.chat.completions.create(
                model="moonshot-v1-128k-vision-preview",
                messages=messages,
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"图像分析失败: {str(e)}"

    def process_text(self, text: str, deep_search: bool = False) -> str:
        """处理文本查询"""
        messages = [self.system_message, {"role": "user", "content": text}]
        
        if deep_search:
            rag_result, picture_paths = call_rag_query(text)
            if rag_result:
                messages.append({"role": "system", "content": f"相关知识：\n{rag_result}"})
        
        completion = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=messages,
            temperature=0.3,
            tools=tools
        )
        
        choice = completion.choices[0]
        if choice.finish_reason == "tool_calls" and hasattr(choice.message, 'tool_calls'):
            messages.append(choice.message)
            
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                if tool_name not in self.tool_map:
                    continue
                
                tool_args = json.loads(tool_call.function.arguments)
                tool_result = self.tool_map[tool_name](tool_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result)
                })
            
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3
            )
        
        response = completion.choices[0].message.content
        if deep_search and picture_paths:
            # 复制相关图片到static目录并获取URL
            image_urls = copy_images_to_static(picture_paths, os.path.dirname(self.upload_folder))
            if image_urls:
                response += "\n\n相关图片："
                for image_info in image_urls:
                    response += f"\n- {image_info['original_name']}"
                return {'text': response, 'files': image_urls}
        return {'text': response, 'files': []}

class LearningHandler(PageHandler):
    def handle_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理learning页面的请求"""
        text = data.get('text', '')
        files = data.get('files', [])
        deep_search = data.get('deep_search', False)
        
        if files and len(files) > 0:
            response_text = self.process_image(files[0]['url'], text)
            if files:
                response_text += "\n文件:\n" + "\n".join([f"- {file['original_name']}" for file in files])
            return {'text': response_text, 'files': files}
        else:
            response = self.process_text(text, deep_search)
            if isinstance(response, dict):
                return response
            return {'text': response, 'files': []}

class UsimageHandler(PageHandler):
    def handle_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理usimage页面的请求"""
        text = data.get('text', '')
        files = data.get('files', [])
        
        if files and len(files) > 0:
            response_text = self.process_image(files[0]['url'], text)
        elif text:
            response_text = self.process_text(text)
        else:
            response_text = "请提供超声图像或相关问题"
        
        return {'text': response_text, 'files': files}