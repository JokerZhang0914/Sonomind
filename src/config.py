import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
API_KEY = os.getenv("API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 语音配置
VOICE_NAME = "zh-CN-XiaoxiaoNeural"

# API URL配置
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/medical/ultrasound"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"