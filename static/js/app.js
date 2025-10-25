class OCRAssistant {
    constructor() {
        this.selectedFiles = [];
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');
        const exportBtn = document.getElementById('exportBtn');

        // 点击上传区域
        uploadArea.addEventListener('click', () => fileInput.click());

        // 文件选择
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));

        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFileSelect(e.dataTransfer.files);
        });

        // 开始识别
        uploadBtn.addEventListener('click', () => this.startOCR());

        // 导出文件
        exportBtn.addEventListener('click', () => this.exportFile());
        

    }

    handleFileSelect(files) {
        this.selectedFiles = Array.from(files).filter(file => 
            file.type.startsWith('image/')
        );

        this.displayFileList();
        
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.disabled = this.selectedFiles.length === 0;
    }

    displayFileList() {
        const fileItems = document.getElementById('fileItems');
        fileItems.innerHTML = '';

        this.selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';

            // 创建图片预览
            const img = document.createElement('img');
            const reader = new FileReader();
            reader.onload = (e) => img.src = e.target.result;
            reader.readAsDataURL(file);

            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            fileInfo.innerHTML = `
                <div class="file-name">${file.name}</div>
                <div class="file-size">${this.formatFileSize(file.size)}</div>
            `;

            fileItem.appendChild(img);
            fileItem.appendChild(fileInfo);
            fileItems.appendChild(fileItem);
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async startOCR() {
        if (this.selectedFiles.length === 0) return;

        const progressSection = document.getElementById('progressSection');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const textEditor = document.getElementById('textEditor');

        progressSection.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '开始识别...';

        const formData = new FormData();
        this.selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        try {
            progressFill.style.width = '30%';
            progressText.textContent = '上传文件中...';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            progressFill.style.width = '70%';
            progressText.textContent = '识别文字中...';

            if (!response.ok) {
                throw new Error('上传失败');
            }

            const result = await response.json();
            
            progressFill.style.width = '100%';
            progressText.textContent = '识别完成！';

            // 合并所有识别结果
            const combinedText = result.results
                .map(item => item.processed_text)
                .join('\n\n---\n\n');

            textEditor.value = combinedText;

            // 3秒后隐藏进度条
            setTimeout(() => {
                progressSection.style.display = 'none';
            }, 3000);

        } catch (error) {
            progressText.textContent = `识别失败: ${error.message}`;
            progressFill.style.width = '0%';
            console.error('OCR Error:', error);
        }
    }



    async exportFile() {
        const textEditor = document.getElementById('textEditor');
        const text = textEditor.value.trim();
        
        if (!text) {
            alert('没有可导出的内容');
            return;
        }

        const format = document.querySelector('input[name="format"]:checked').value;
        const filename = document.getElementById('filename').value || 
                        `ocr_result_${new Date().getTime()}`;

        try {
            const response = await fetch('/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    format: format,
                    filename: filename
                })
            });

            if (!response.ok) {
                throw new Error('导出失败');
            }

            // 下载文件
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${filename}.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

        } catch (error) {
            alert(`导出失败: ${error.message}`);
            console.error('Export Error:', error);
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new OCRAssistant();
});