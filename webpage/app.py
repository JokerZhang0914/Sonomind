from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
from uuid import uuid4
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 导入自定义服务模块
from src.chat.chat_service import tool_map as chat_tool_map


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

# 导入页面处理器
from page_handlers import LearningHandler, UsimageHandler

# 初始化页面处理器
learning_handler = LearningHandler(app.config['UPLOAD_FOLDER'])
usimage_handler = UsimageHandler(app.config['UPLOAD_FOLDER'])

# 处理文本和文件的端点
@app.route('/ask', methods=['POST'])
def ask():
    # 获取JSON请求数据
    data = request.get_json()
    page_type = data.get('page_type', 'learning')  # 获取页面类型，默认为learning

    # 根据页面类型选择处理器
    if page_type == 'learning':
        response = learning_handler.handle_request(data)
    elif page_type == 'usimage':
        response = usimage_handler.handle_request(data)
    else:
        response = {'text': '无效的页面类型', 'files': []}

    # 返回响应
    return jsonify(response)

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