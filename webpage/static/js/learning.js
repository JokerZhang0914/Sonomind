// 获取DOM元素
const chatContainer = document.getElementById('chatContainer'); // 聊天消息容器
const inputContainer = document.getElementById('inputContainer'); // 输入框容器
const chatInput = document.getElementById('chatInput'); // 文本输入框
const clipIcon = document.getElementById('clipIcon'); // 附件图标
const sendIcon = document.getElementById('sendIcon'); // 发送图标
const fileInput = document.getElementById('fileInput'); // 文件输入框
const filePreviewContainer = document.getElementById('filePreviewContainer'); // 文件预览容器

// 定义变量
let selectedFiles = []; // 存储用户选择的文件对象数组
let nextId = 0; // 为每个文件分配唯一ID

// 创建文件预览的DOM元素（输入框上方的预览）
function createFilePreview(item) {
    const filePreview = document.createElement('div'); // 创建预览容器
    filePreview.className = 'file-preview'; // 设置CSS类
    filePreview.innerHTML = `
        <span>${item.file.name}</span>
        <button class="delete-btn">×</button>
    `; // 显示文件名和删除按钮
    // 为删除按钮添加点击事件
    filePreview.querySelector('.delete-btn').addEventListener('click', () => {
        selectedFiles = selectedFiles.filter(i => i.id !== item.id); // 移除文件
        updatePreviews(); // 更新预览
    });
    return filePreview;
}

// 创建文件展示的DOM元素（用于消息中）
function createFileDisplay(fileInfo, isImagePreview = false) {
    const fileDisplay = document.createElement('div'); // 创建展示容器
    fileDisplay.className = 'file-preview'; // 使用相同样式
    // 检查文件是否为图片（仅对后端响应中的文件启用图片预览）
    const isImage = isImagePreview && /\.(png|jpg|jpeg)$/i.test(fileInfo.original_name);
    if (isImage) {
        // 如果是图片，直接展示
        const img = document.createElement('img');
        img.src = fileInfo.url;
        img.style.maxWidth = '200px'; // 限制图片宽度
        fileDisplay.appendChild(img);
    } else {
        // 非图片文件，显示可点击的文件名
        const fileLink = document.createElement('a');
        fileLink.href = fileInfo.url; // 设置文件URL
        fileLink.textContent = fileInfo.original_name; // 显示原始文件名
        fileLink.style.color = '#007bff'; // 链接颜色
        fileLink.style.textDecoration = 'underline'; // 添加下划线
        fileLink.target = '_blank'; // 在新标签页打开
        fileDisplay.appendChild(fileLink);
    }
    return fileDisplay;
}

// 更新文件预览区域
function updatePreviews() {
    filePreviewContainer.innerHTML = ''; // 清空预览容器
    selectedFiles.forEach(item => {
        const filePreview = createFilePreview(item); // 创建预览
        filePreviewContainer.appendChild(filePreview); // 添加到容器
    });
}

// 异步函数：上传文件到服务器
async function uploadFiles(files) {
    const formData = new FormData(); // 创建FormData对象
    for (const file of files) {
        formData.append('files', file); // 添加文件
    }
    // 发送POST请求到/upload
    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    if (response.ok) {
        return result.files; // 返回文件元数据数组（包含original_name和url）
    } else {
        throw new Error(result.error || '文件上传失败');
    }
}

// 异步函数：发送消息
async function sendMessage() {
    const userMessage = chatInput.value.trim(); // 获取用户输入
    const userFiles = selectedFiles.map(item => item.file); // 获取文件

    // 如果没有消息和文件，直接返回
    if (!userMessage && userFiles.length === 0) {
        return;
    }

    try {
        // 显示用户消息
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        if (userMessage) {
            userDiv.textContent = userMessage; // 显示文本
            chatContainer.appendChild(userDiv);
        }

        // 上传文件并显示（仅显示文件名）
        let uploadedFiles = [];
        if (userFiles.length > 0) {
            uploadedFiles = await uploadFiles(userFiles); // 获取文件元数据
            const fileDisplayContainer = document.createElement('div');
            fileDisplayContainer.className = 'file-preview-container';
            fileDisplayContainer.style.marginTop = '5px';
            fileDisplayContainer.style.alignSelf = 'flex-end';
            uploadedFiles.forEach(fileInfo => {
                const fileDisplay = createFileDisplay(fileInfo, false); // 用户消息不预览图片
                fileDisplayContainer.appendChild(fileDisplay);
            });
            chatContainer.appendChild(fileDisplayContainer);
        }

        // 发送文本和文件元数据到/ask
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: userMessage,
                files: uploadedFiles,
                page_type: 'learning' // 指定页面类型为learning
            })
        });
        const result = await response.json();

        // 显示后端响应
        const responseDiv = document.createElement('div');
        responseDiv.className = 'message response';
        responseDiv.innerHTML = marked.parse(result.text); // 渲染Markdown
        if (result.files && result.files.length > 0) {
            const fileDisplayContainer = document.createElement('div');
            fileDisplayContainer.className = 'file-preview-container';
            fileDisplayContainer.style.marginTop = '5px';
            result.files.forEach(fileInfo => {
                const fileDisplay = createFileDisplay(fileInfo, true); // 后端响应中图片直接展示
                fileDisplayContainer.appendChild(fileDisplay);
            });
            responseDiv.appendChild(fileDisplayContainer);
        }
        chatContainer.appendChild(responseDiv);

        // 清空输入和文件
        chatInput.value = '';
        selectedFiles = [];
        filePreviewContainer.innerHTML = '';
        fileInput.value = '';

        // 调整输入框高度
        chatInput.style.height = 'auto';
        chatInput.style.height = `${chatInput.scrollHeight}px`;

        // 移除居中样式
        inputContainer.classList.remove('centered');

        // 滚动到底部
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (error) {
        console.error('错误:', error);
        alert('操作失败：' + error.message);
    }
}

// 监听输入框变化，动态调整高度
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = `${this.scrollHeight}px`;
});

// 监听Enter键发送消息
chatInput.addEventListener('keydown', async function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        await sendMessage();
    }
});

// 监听发送图标点击
sendIcon.addEventListener('click', sendMessage);

// 监听附件图标点击
clipIcon.addEventListener('click', function() {
    fileInput.click();
});

// 监听文件选择变化
fileInput.addEventListener('change', function() {
    const newFiles = Array.from(fileInput.files);
    newFiles.forEach(file => {
        selectedFiles.push({ id: nextId++, file });
    });
    fileInput.value = '';
    updatePreviews();
});