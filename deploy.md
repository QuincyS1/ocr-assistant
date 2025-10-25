# OCR助手部署指南

## 方案1：本地网络共享（推荐）

### 步骤：
1. 启动应用：`python app.py`
2. 获取本机IP：`ifconfig | grep inet`
3. 其他人访问：`http://你的IP:5005`

### 优点：
- 简单快速
- 无需额外配置
- 适合局域网使用

## 方案2：云服务器部署

### 使用Railway（免费）：
1. 注册 https://railway.app
2. 连接GitHub仓库
3. 自动部署

### 使用Heroku：
1. 安装Heroku CLI
2. `heroku create ocr-assistant`
3. `git push heroku main`

## 方案3：Docker容器

### 创建Dockerfile：
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5005
CMD ["python", "app.py"]
```

### 运行：
```bash
docker build -t ocr-assistant .
docker run -p 5005:5005 ocr-assistant
```

## 方案4：ngrok隧道（临时）

### 安装ngrok：
```bash
brew install ngrok
```

### 使用：
```bash
python app.py  # 终端1
ngrok http 5005  # 终端2
```

分享ngrok提供的公网URL即可。

## 推荐顺序：
1. 本地网络 → 2. ngrok → 3. 云服务器 → 4. Docker