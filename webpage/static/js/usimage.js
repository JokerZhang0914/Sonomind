// 获取DOM元素
const interactionContainer = document.getElementById('interactionContainer'); // 交互容器
const uploadContainer = document.getElementById('uploadContainer'); // 上传区域容器
const uploadBox = document.getElementById('uploadBox'); // 上传方框
const plusIcon = document.getElementById('plusIcon'); // +号图片
const fileInput = document.getElementById('fileInput'); // 文件输入框
const resetButton = document.getElementById('resetButton'); // 重置按钮
const sendButton = document.getElementById('sendButton'); // 发送按钮

// 定义变量
let selectedFile = null; // 存储用户选择的图片文件
let isFileUploaded = false; // 标记是否已上传图片，防止重复上传

// 创建文件展示的DOM元素（用于交互容器中）
function createFileDisplay(fileInfo, isImagePreview = false) {
    const fileDisplay = document.createElement('div'); // 创建展示容器
    fileDisplay.className = 'file-preview'; // 设置CSS类
    const isImage = isImagePreview && /\.(png|jpg|jpeg|gif|webp)$/i.test(fileInfo.original_name);
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

// 异步函数：发送图片
async function sendImage() {
    if (!selectedFile) {
        alert('请先上传图片'); // 未选择图片时提示
        return;
    }
    try {
        // 上传图片
        const uploadedFiles = await uploadFiles([selectedFile]);
        // 显示交互容器
        interactionContainer.style.display = 'block';
        // 隐藏上传区域
        uploadContainer.style.display = 'none';
        // 显示用户上传的图片（居中图片）
        const userFileDiv = document.createElement('div');
        userFileDiv.className = 'message user';
        uploadedFiles.forEach(fileInfo => {
            const img = document.createElement('img');
            img.src = fileInfo.url;
            img.style.maxWidth = '200px'; // 限制图片宽度
            userFileDiv.appendChild(img);
        });
        interactionContainer.appendChild(userFileDiv);
        // 发送到/ask
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: '', // 无文本内容
                files: uploadedFiles,
                page_type: 'usimage' // 指定页面类型为usimage
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
            result.files.forEach(fileInfo => {
                const fileDisplay = createFileDisplay(fileInfo, true); // 后端响应中图片直接展示
                fileDisplayContainer.appendChild(fileDisplay);
            });
            responseDiv.appendChild(fileDisplayContainer);
        }
        interactionContainer.appendChild(responseDiv);
        // 清空选择的文件
        selectedFile = null;
        isFileUploaded = false;
        fileInput.value = '';
        // 滚动到底部
        interactionContainer.scrollTop = interactionContainer.scrollHeight;
    } catch (error) {
        console.error('错误:', error);
        alert('操作失败：' + error.message);
    }
}

// 事件监听器
uploadBox.addEventListener('click', () => {
    if (!isFileUploaded) {
        fileInput.click(); // 未上传时触发文件选择
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        selectedFile = fileInput.files[0]; // 存储选择的文件
        isFileUploaded = true; // 标记已上传，禁用重复上传
        // 显示图片预览，仅替换加号
        const reader = new FileReader();
        reader.onload = (e) => {
            plusIcon.src = e.target.result; // 替换加号图片
            plusIcon.style.maxWidth = '100%'; // 确保图片适应方框
            plusIcon.style.maxHeight = '70%'; // 限制图片高度，留出文字空间
            uploadBox.style.cursor = 'default'; // 移除手型光标
        };
        reader.readAsDataURL(selectedFile);
    }
});

resetButton.addEventListener('click', () => {
    location.reload(); // 点击重置按钮重新加载页面
}); // 重置页面

sendButton.addEventListener('click', sendImage); // 发送图片

// 初始化滚动
interactionContainer.scrollTop = interactionContainer.scrollHeight;