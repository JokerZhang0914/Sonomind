from typing import *
import os
import base64

def normalize_path_separator(path: str) -> str:
    """统一处理路径分隔符，支持Windows(\)和Python(/)风格"""
    path = path.replace('"', '')  
    path.replace('\\', '/') # 将所有的 / 替换为 \
    return path

def normalize_path(path: str) -> str:
    """规范化路径格式，处理路径分隔符并确保路径有效"""
    # 统一分隔符
    path = normalize_path_separator(path)
    # 使用os.path.normpath处理 . 和 .. 等特殊路径
    return os.path.normpath(path)

def encode_image_to_base64(image_path: str) -> str:
    """将图像文件编码为base64格式"""
    # 规范化路径
    image_path = normalize_path(image_path)
    
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    file_ext = os.path.splitext(image_path)[1][1:]  # 获取扩展名（去掉点）
    if not file_ext:
        file_ext = "png"  # 默认扩展名
    
    return f"data:image/{file_ext};base64,{base64.b64encode(image_data).decode('utf-8')}"

def has_image_content(messages: List[Dict]) -> bool:
    """检查消息列表中是否包含图像内容"""
    return any(
        isinstance(msg.get("content"), list) and 
        any(item.get("type") == "image_url" for item in msg.get("content", []))
        for msg in messages if isinstance(msg, dict)
    )