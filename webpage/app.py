from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
from uuid import uuid4

# 初始化Flask应用
app = Flask(__name__)

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

    # 响应文本
    response_text = ""

    # 先判断页面类型
    if page_type == 'learning':
        # 如果是learning页面，再判断是否需要深度搜索
        if deep_search:
            # 深度搜索逻辑：后续替换为实际的处理逻辑
            response_text = f"学习页面深度搜索收到消息：{text}"
            if files:
                response_text += "\n文件:\n" + "\n".join([f"- {file['original_name']}" for file in files])
        else:
            # 普通逻辑：后续替换为实际的处理逻辑
            response_text = f"学习页面收到消息：{text}"
            if files:
                response_text += "\n文件:\n" + "\n".join([f"- {file['original_name']}" for file in files])
    elif page_type == 'usimage':
        # usimage页面逻辑：后续替换为实际的处理逻辑
        response_text = "影像分析页面收到图片："
        if files:
            response_text += "\n" + "\n".join([f"- {file['original_name']}" for file in files])
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