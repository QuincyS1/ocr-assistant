# 字体文件说明

## 中文字体支持

为了在PDF中正确显示中文，请将以下字体文件放置在此目录：

### 推荐字体文件：
1. `NotoSansCJKsc-Regular.ttf` - Google Noto字体（推荐）
2. `NotoSansCJKsc-Regular.otf` - OpenType版本
3. `SourceHanSansSC-Regular.otf` - Adobe思源黑体

### 下载地址：
- Noto字体：https://fonts.google.com/noto/specimen/Noto+Sans+SC
- 思源字体：https://github.com/adobe-fonts/source-han-sans

### 使用说明：
1. 下载任意一个中文字体文件
2. 重命名为 `NotoSansCJKsc-Regular.ttf` 或 `NotoSansCJKsc-Regular.otf`
3. 放置在此 `fonts/` 目录下
4. 重启Flask应用

### 备用方案：
如果没有字体文件，系统会自动尝试使用macOS系统字体：
- PingFang.ttc
- Hiragino Sans GB.ttc

### 验证：
生成PDF后，如果中文显示正常（非方块），说明字体配置成功。