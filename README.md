# 多图OCR识别与文本整合助手

一个基于Flask的Web应用，支持批量图片OCR识别、智能文本清理和多格式导出。

## 功能特点

- 📸 批量图片上传和OCR识别
- 🤖 使用Google Gemini 2.5 Flash进行文字识别
- 🧹 智能文本清理和去重
- 📄 支持TXT、DOCX、Markdown、PDF格式导出
- 🔗 自动检测和合并连续截图
- 🌐 支持中英文混排

## 在线使用

访问：[https://your-app.vercel.app](https://your-app.vercel.app)

## 本地运行

1. 克隆项目：
```bash
git clone <your-repo-url>
cd ocr-assistant
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置API密钥：
编辑 `app.py` 中的 `GEMINI_API_KEY`

4. 运行应用：
```bash
python app.py
```

5. 访问：http://localhost:5005

## 部署到Vercel

1. Fork此项目到你的GitHub
2. 在Vercel中导入项目
3. 设置环境变量 `GEMINI_API_KEY`
4. 部署完成

## 技术栈

- 后端：Flask + Python
- OCR：Google Gemini 2.5 Flash
- PDF生成：ReportLab
- 前端：HTML + CSS + JavaScript
- 部署：Vercel

## 许可证

MIT License