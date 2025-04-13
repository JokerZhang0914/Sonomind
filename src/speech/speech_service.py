from typing import *
import asyncio
import edge_tts
import pygame
import io
from ..config import VOICE_NAME

async def text_to_speech(text: str) -> Dict[str, Any]:
    """使用EdgeTTS将文本转换为语音并直接播放"""
    try:
        # 初始化pygame音频系统
        pygame.mixer.init()
        
        # 创建TTS通信对象
        communicate = edge_tts.Communicate(text, VOICE_NAME)
        
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