from flask import Flask, request, jsonify, render_template, send_file
import os
import base64
import requests
import json
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import inch
import uuid
import re
from difflib import SequenceMatcher
from PIL import Image
import io

app = Flask(__name__)

# Vercel兼容性
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Gemini API配置
GEMINI_API_KEY = "AIzaSyDeYj-tpMueljip6MnfjGNjDggllevLjwY"

class OCRProcessor:
    def __init__(self):
        print("初始化OCR处理器")
    
    def ocr_image(self, image_data):
        # 使用Gemini API进行OCR识别
        gemini_result = self._try_gemini_ocr(image_data)
        if not gemini_result.startswith("API错误") and not gemini_result.startswith("OCR识别错误"):
            return gemini_result
        
        # Gemini失败时返回提示信息
        return "图片识别失败。请检查网络连接或API配置。您也可以手动输入文本内容进行整理和导出。"
    
    def _try_gemini_ocr(self, image_data):
        if not GEMINI_API_KEY or GEMINI_API_KEY == 'your_gemini_api_key':
            return "API错误: 未配置Gemini API密钥"
        
        img_base64 = base64.b64encode(image_data).decode('utf-8')
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "请识别图片中的所有文字。重要要求：1)保持原图片的段落结构，段落之间用双换行分隔。2)将同一段落内的换行合并为连续文本，英文单词间加空格，中文字符直接连接。3)只输出整理后的文字内容，不要添加任何说明。"
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                return f"API错误: HTTP {response.status_code}"
            
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                print(f"Gemini识别成功")
                return text
            elif 'error' in result:
                return f"API错误: {result['error']['message']}"
            return "API错误: 未识别到文字"
        except Exception as e:
            return f"OCR识别错误: {str(e)}"
    


class TextProcessor:

    
    @staticmethod
    def clean_noise_text(text):
        import re
        
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 只删除明显的广告词
            if any(keyword in line for keyword in ["下载应用", "立即下载", "扫码下载", "点击下载", "立即加入", "免费会员", "开通会员"]):
                continue
            
            # 删除时间戳 (如: 2023-10-26 13:40:12, 13:40, 10月26日)
            if re.match(r'^\d{1,4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?\s*\d{1,2}[时:]\d{1,2}', line):
                continue
            if re.match(r'^\d{1,2}[时:]\d{1,2}', line) and len(line) < 10:
                continue
            
            # 删除页码 (如: 第1页, 1/10, Page 1)
            if re.match(r'^(第\d+页|\d+/\d+|Page\s*\d+)$', line):
                continue
            
            clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    @staticmethod
    def merge_ocr_results(results):
        """改进的OCR结果合并，支持长文本重复检测和去重"""
        if len(results) <= 1:
            return results[0]['processed_text'] if results else ""
        
        # 提取所有页面文本
        pages = [result['processed_text'] for result in results]
        
        # 清理每页文本
        cleaned_pages = []
        for page in pages:
            # 移除中文间空格
            cleaned = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', page)
            cleaned_pages.append(cleaned.strip())
        
        # 检测和合并重复内容
        merged_pages = TextProcessor.detect_and_merge_overlaps(cleaned_pages)
        
        # 最终清理
        return TextProcessor.final_cleanup(merged_pages)
    
    @staticmethod
    def detect_and_merge_overlaps(pages):
        """检测并合并重复内容"""
        if len(pages) <= 1:
            return '\n\n'.join(pages)
        
        # 构建重复关系图
        overlap_graph = {}
        for i in range(len(pages)):
            for j in range(len(pages)):
                if i != j:
                    overlap_info = TextProcessor.find_overlap(pages[i], pages[j])
                    if overlap_info['has_overlap']:
                        overlap_graph[i] = {'target': j, 'info': overlap_info}
        
        # 按顺序合并
        merged_content = []
        used_pages = set()
        
        for i in range(len(pages)):
            if i in used_pages:
                continue
                
            current_page = pages[i]
            used_pages.add(i)
            
            # 查找与当前页重复的页面
            if i in overlap_graph:
                target_idx = overlap_graph[i]['target']
                overlap_info = overlap_graph[i]['info']
                
                if target_idx not in used_pages:
                    # 合并两页
                    merged_page = TextProcessor.merge_two_overlapping_pages(
                        current_page, pages[target_idx], overlap_info
                    )
                    merged_content.append(merged_page)
                    used_pages.add(target_idx)
                else:
                    merged_content.append(current_page)
            else:
                merged_content.append(current_page)
        
        return '\n\n'.join(merged_content)
    
    @staticmethod
    def find_overlap(page1, page2):
        """使用SequenceMatcher检测两页间的重复"""
        # 将文本分解为句子
        sentences1 = re.split(r'[\u3002！？.!?]', page1)
        sentences2 = re.split(r'[\u3002！？.!?]', page2)
        
        # 清理空句子
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # 检测最长公共子序列
        matcher = SequenceMatcher(None, sentences1, sentences2)
        match = matcher.find_longest_match(0, len(sentences1), 0, len(sentences2))
        
        if match.size > 0:
            # 计算重复内容的字符数
            overlap_sentences = sentences1[match.a:match.a + match.size]
            overlap_text = ''.join(overlap_sentences)
            
            # 判断是否为有意义的重复
            if len(overlap_text) >= 20 and match.size >= 1:
                return {
                    'has_overlap': True,
                    'page1_start': match.a,
                    'page1_end': match.a + match.size,
                    'page2_start': match.b,
                    'page2_end': match.b + match.size,
                    'overlap_length': len(overlap_text)
                }
        
        return {'has_overlap': False}
    
    @staticmethod
    def merge_two_overlapping_pages(page1, page2, overlap_info):
        """合并两个有重复的页面"""
        sentences1 = re.split(r'[\u3002！？.!?]', page1)
        sentences2 = re.split(r'[\u3002！？.!?]', page2)
        
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # 取page1的非重复部分 + page2的非重复部分
        result_sentences = []
        
        # 添加page1的前部分
        result_sentences.extend(sentences1[:overlap_info['page1_end']])
        
        # 添加page2的后部分（去除重复部分）
        result_sentences.extend(sentences2[overlap_info['page2_end']:])
        
        # 重新组合成段落
        return '。'.join(result_sentences) + '。' if result_sentences else ''
    
    @staticmethod
    def final_cleanup(text):
        """最终清理合并后的文本 - 保持段落结构"""
        if not text:
            return ""
        
        # 移除中文间空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # 按段落处理（保持双换行分隔）
        paragraphs = text.split('\n\n')
        clean_paragraphs = []
        seen_paragraphs = set()
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and paragraph not in seen_paragraphs:
                clean_paragraphs.append(paragraph)
                seen_paragraphs.add(paragraph)
        
        return '\n\n'.join(clean_paragraphs)
    
    @staticmethod
    def is_continuous(text1, text2):
        """判断两页是否连续"""
        lines1 = [line.strip() for line in text1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in text2.split('\n') if line.strip()]
        
        if not lines1 or not lines2:
            return False
        
        # 检查text2的前两行是否在text1中出现
        for line2 in lines2[:2]:
            for line1 in lines1:
                if line2 in line1 or line1 in line2:
                    return True
        
        # 检查text1的前两行是否在text2中出现
        for line1 in lines1[:2]:
            for line2 in lines2:
                if line1 in line2 or line2 in line1:
                    return True
        
        return False
    
    @staticmethod
    def reorder_pages(connections, n):
        """根据连续性关系重排序页面"""
        if not connections:
            return list(range(n))
        
        # 找到起始页（没有被其他页指向的页）
        pointed_to = set(connections.values())
        start_pages = [i for i in range(n) if i not in pointed_to]
        
        if not start_pages:
            start_pages = [0]  # 默认从第一页开始
        
        # 按连续性链排序
        ordered = []
        visited = set()
        
        for start in start_pages:
            current = start
            while current is not None and current not in visited:
                ordered.append(current)
                visited.add(current)
                current = connections.get(current)
        
        # 添加未访问的页面
        for i in range(n):
            if i not in visited:
                ordered.append(i)
        
        return ordered
    
    @staticmethod
    def smart_merge(text1, text2):
        """智能合并两段文本，去除重复"""
        lines1 = [line.strip() for line in text1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in text2.split('\n') if line.strip()]
        
        # 找到重复的起始位置
        start_idx = 0
        for i, line2 in enumerate(lines2):
            found_overlap = False
            for line1 in lines1[-3:]:  # 检查text1的最后3行
                if line2 in line1 or line1 in line2:
                    start_idx = i + 1
                    found_overlap = True
                    break
            if not found_overlap:
                break
        
        return '\n'.join(lines2[start_idx:])
    
    @staticmethod
    def clean_merged_text(text):
        """清理合并后的文本，去除重复行和空行"""
        lines = text.split('\n')
        clean_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                clean_lines.append(line)
                seen_lines.add(line)
        
        # 重新组织段落
        paragraphs = []
        current_paragraph = []
        
        for line in clean_lines:
            if line.endswith(('。', '.', '!', '?', '！', '？')):
                current_paragraph.append(line)
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    @staticmethod
    def find_text_overlap(text1, text2):
        """计算两段文本的重复率"""
        lines1 = text1.split('\n')[-3:]  # 取最后3行
        lines2 = text2.split('\n')[:3]   # 取前3行
        
        overlap_count = 0
        total_lines = len(lines2)
        
        for line2 in lines2:
            for line1 in lines1:
                if line2.strip() and line1.strip() and line2.strip() in line1.strip():
                    overlap_count += 1
                    break
        
        return overlap_count / total_lines if total_lines > 0 else 0
    
    @staticmethod
    def remove_overlap(text1, text2):
        """从第二段文本中移除与第一段重复的部分"""
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        # 找到重复的起始位置
        start_idx = 0
        for i, line2 in enumerate(lines2):
            if any(line2.strip() in line1 for line1 in lines1[-3:] if line1.strip()):
                start_idx = i + 1
            else:
                break
        
        return '\n'.join(lines2[start_idx:])
    
    @staticmethod
    def llm_content_filter(text):
        """使用Gemini进一步过滤非正文内容"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"请从以下文本中提取正文内容，去除广告、按钮、导航等非正文元素，只保留有意义的文章内容：\n\n{text}"
                }]
            }]
        }
        
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        except:
            pass
        return text
    
    @staticmethod
    def remove_chinese_spaces(text):
        """移除中文字符之间的多余空格"""
        import re
        # 移除中文字符之间的空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        # 移除中文与标点之间的空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[，。！？；：])', '', text)
        text = re.sub(r'(?<=[，。！？；：])\s+(?=[\u4e00-\u9fff])', '', text)
        return text
    
    @staticmethod
    def clean_and_merge_texts(pages):
        """整合空格清理与多页拼接的主函数"""
        if not pages:
            return ""
        
        # 1. 对每页进行基础清理和空格处理
        cleaned_pages = []
        for page_text in pages:
            # 基础噪声清理
            cleaned = TextProcessor.clean_noise_text(page_text)
            # 移除中文间多余空格
            cleaned = TextProcessor.remove_chinese_spaces(cleaned)
            if cleaned.strip():
                cleaned_pages.append(cleaned)
        
        if not cleaned_pages:
            return ""
        
        if len(cleaned_pages) == 1:
            return TextProcessor.format_final_text(cleaned_pages[0])
        
        # 2. 检测页面连续性并重排序
        ordered_pages = TextProcessor.detect_and_reorder_pages(cleaned_pages)
        
        # 3. 智能拼接去重
        merged_text = TextProcessor.smart_merge_pages(ordered_pages)
        
        # 4. 最终格式化
        return TextProcessor.format_final_text(merged_text)
    
    @staticmethod
    def detect_and_reorder_pages(pages):
        """检测页面连续性并重新排序"""
        n = len(pages)
        if n <= 1:
            return pages
        
        # 构建连续性关系图
        connections = {}
        for i in range(n):
            for j in range(n):
                if i != j and TextProcessor.pages_are_continuous(pages[i], pages[j]):
                    connections[i] = j
        
        if not connections:
            return pages  # 无连续关系，保持原顺序
        
        # 找到起始页面（没有被其他页面指向）
        pointed_to = set(connections.values())
        start_pages = [i for i in range(n) if i not in pointed_to]
        
        if not start_pages:
            start_pages = [0]  # 如果形成环，从第一页开始
        
        # 按连续性链重排序
        ordered_indices = []
        visited = set()
        
        for start in start_pages:
            current = start
            while current is not None and current not in visited:
                ordered_indices.append(current)
                visited.add(current)
                current = connections.get(current)
        
        # 添加未访问的页面
        for i in range(n):
            if i not in visited:
                ordered_indices.append(i)
        
        return [pages[i] for i in ordered_indices]
    
    @staticmethod
    def pages_are_continuous(page1, page2):
        """判断两页是否连续"""
        lines1 = [line.strip() for line in page1.split('\n') if line.strip()]
        lines2 = [line.strip() for line in page2.split('\n') if line.strip()]
        
        if not lines1 or not lines2:
            return False
        
        # 检查page2的前两行是否在page1的任意行中完全匹配
        for line2 in lines2[:2]:
            if any(line2 == line1 for line1 in lines1):
                return True
        
        # 检查page1的前两行是否在page2的任意行中完全匹配
        for line1 in lines1[:2]:
            if any(line1 == line2 for line2 in lines2):
                return True
        
        return False
    
    @staticmethod
    def smart_merge_pages(pages):
        """智能拼接多页，去除重复内容"""
        if not pages:
            return ""
        
        merged_text = pages[0]
        
        for i in range(1, len(pages)):
            current_page = pages[i]
            merged_text = TextProcessor.merge_two_pages(merged_text, current_page)
        
        return merged_text
    
    @staticmethod
    def merge_two_pages(text1, text2):
        """合并两页文本，去除重复部分"""
        # 按句子分割来检测重复
        sentences1 = re.split(r'[。！？.!?]', text1)
        sentences2 = re.split(r'[。！？.!?]', text2)
        
        # 清理空句子
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # 找到重复的起始位置
        overlap_start = 0
        for i, sent2 in enumerate(sentences2):
            # 检查这个句子是否在text1的后几个句子中出现
            found_in_text1 = False
            for sent1 in sentences1[-3:]:
                if sent2 in sent1 or sent1 in sent2 or sent2 == sent1:
                    found_in_text1 = True
                    break
            
            if found_in_text1:
                overlap_start = i + 1
            else:
                break
        
        # 拼接非重复部分
        unique_sentences2 = sentences2[overlap_start:]
        if unique_sentences2:
            # 重新组合成段落
            all_sentences = sentences1 + unique_sentences2
            return '。'.join(all_sentences) + '。' if all_sentences else ''
        else:
            return '。'.join(sentences1) + '。' if sentences1 else ''
    
    @staticmethod
    def format_final_text(text):
        """最终文本格式化处理 - 智能合并换行保持段落"""
        if not text:
            return ""
        
        # 先移除中文间多余空格
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # 按双换行分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # 对每个段落内部处理
            lines = paragraph.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                # 智能合并：英文加空格，中文直接连接
                merged_text = ""
                for i, line in enumerate(cleaned_lines):
                    if i == 0:
                        merged_text = line
                    else:
                        # 如果前一行以英文结尾且当前行以英文开头，加空格
                        prev_line = cleaned_lines[i-1]
                        if (re.search(r'[a-zA-Z]$', prev_line) and 
                            re.search(r'^[a-zA-Z]', line)):
                            merged_text += " " + line
                        else:
                            merged_text += line
                
                formatted_paragraphs.append(merged_text)
        
        return '\n\n'.join(formatted_paragraphs)
    
    @staticmethod
    def clean_and_format(text):
        """单页文本清理和格式化 - 强制段落处理"""
        if not text:
            return ""
        
        # 先进行基础清理
        cleaned_text = TextProcessor.clean_noise_text(text)
        
        # 强制段落处理
        return TextProcessor.force_paragraph_formatting(cleaned_text)
    
    @staticmethod
    def force_paragraph_formatting(text):
        """强制段落格式化 - 智能检测段落"""
        if not text:
            return ""
        
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # 空行表示段落结束
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                # 智能检测段落分隔
                should_start_new_paragraph = False
                
                if current_paragraph:
                    prev_line = current_paragraph[-1]
                    
                    # 检测段落结束标志
                    if (prev_line.endswith(('。', '.', '!', '?', '！', '？')) and
                        not line.startswith(('但是', '然而', '因此', '所以', 'However', 'But', 'Therefore'))):
                        should_start_new_paragraph = True
                    
                    # 检测缩进或特殊格式
                    if (line.startswith(('    ', '\t', '　　')) or
                        re.match(r'^\d+[\.）]', line) or  # 编号列表
                        re.match(r'^[A-Za-z]\)', line)):   # 字母列表
                        should_start_new_paragraph = True
                
                if should_start_new_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
        
        # 处理最后一个段落
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # 合并每个段落内的行
        formatted_paragraphs = []
        for paragraph_lines in paragraphs:
            if not paragraph_lines:
                continue
            
            # 智能合并行
            merged_text = ""
            for i, line in enumerate(paragraph_lines):
                if i == 0:
                    merged_text = line
                else:
                    # 英文单词间加空格，中文直接连接
                    prev_line = paragraph_lines[i-1]
                    if (re.search(r'[a-zA-Z]$', prev_line) and 
                        re.search(r'^[a-zA-Z]', line)):
                        merged_text += " " + line
                    else:
                        merged_text += line
            
            formatted_paragraphs.append(merged_text)
        
        # 用双换行分隔段落
        return '\n\n'.join(formatted_paragraphs)
    
    @staticmethod
    def llm_semantic_correction(text):
        """使用LLM进行语义修正，保留语义连贯的句子"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"请修正以下文本，保留语义连贯的句子，删除不完整或无意义的片段，使文本更加流畅自然：\n\n{text}"
                }]
            }]
        }
        
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        except:
            pass
        return text

    @staticmethod
    def dev_clean_text(ocr_text: str) -> str:
        """
        🧠 开发者命令：清理OCR换行符，只保留段落空行。
        功能：
        1. 删除段内换行（合并为一行）
        2. 保留段落空行（双换行）
        3. 去掉多余空格
        """
        if not ocr_text:
            return ""
        
        text = ocr_text.replace("\r\n", "\n").strip()
        # 合并行内换行，只保留段落空行
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # 清理多余空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 按双换行分段，去掉段首尾空格
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return "\n\n".join(paragraphs)


class FileExporter:
    @staticmethod
    def export_txt(text, filename):
        filepath = f"/tmp/{filename}.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        return filepath
    
    @staticmethod
    def export_docx(text, filename):
        doc = Document()
        for paragraph in text.split('\n\n'):
            doc.add_paragraph(paragraph)
        
        filepath = f"/tmp/{filename}.docx"
        doc.save(filepath)
        return filepath
    
    @staticmethod
    def export_md(text, filename):
        md_content = text.replace('\n\n', '\n\n')
        filepath = f"/tmp/{filename}.md"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        return filepath
    
    @staticmethod
    def export_pdf(text, filename):
        filepath = f"/tmp/{filename}.pdf"
        
        # 注册中文字体
        font_name = 'NotoSans'
        font_registered = False
        
        # 尝试多个字体路径
        font_paths = [
            'fonts/NotoSansCJK.ttc',
            'fonts/NotoSansCJKsc-Regular.otf',
            'fonts/NotoSansCJKsc-Regular.ttf',
            '/Users/quincys/Library/Fonts/NotoSansCJK.ttc',
            '/opt/homebrew/Caskroom/font-noto-sans-cjk/2.004/NotoSansCJK.ttc',
            'static/fonts/NotoSansCJKsc-Regular.ttf'
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    # 对于.ttc文件，指定使用第一个字体
                    if font_path.endswith('.ttc'):
                        pdfmetrics.registerFont(TTFont(font_name, font_path, subfontIndex=0))
                    else:
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                    font_registered = True
                    print(f"成功注册字体: {font_path}")
                    break
            except Exception as e:
                print(f"注册字体失败 {font_path}: {e}")
                continue
        
        if not font_registered:
            # 尝试使用系统默认中文字体
            try:
                from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                font_name = 'STSong-Light'
                font_registered = True
                print("使用CID字体: STSong-Light")
            except:
                font_name = 'Helvetica'
                print("警告: 未找到中文字体，使用默认字体")
        
        # 创建PDF文档
        doc = SimpleDocTemplate(filepath, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # 创建段落样式
        styles = getSampleStyleSheet()
        chinese_style = ParagraphStyle(
            'ChineseStyle',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=0,
            rightIndent=0
        )
        
        # 清理文本中的多余空格
        cleaned_text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        # 构建PDF内容 - 保持段落结构
        story = []
        
        # 按双换行分割段落
        for paragraph_text in cleaned_text.split('\n\n'):
            if paragraph_text.strip():
                # 转义HTML特殊字符
                paragraph_text = paragraph_text.replace('&', '&amp;')
                paragraph_text = paragraph_text.replace('<', '&lt;')
                paragraph_text = paragraph_text.replace('>', '&gt;')
                
                para = Paragraph(paragraph_text, chinese_style)
                story.append(para)
                story.append(Spacer(1, 12))  # 段落间距
        
        # 生成PDF
        doc.build(story)
        return filepath

ocr_processor = OCRProcessor()
text_processor = TextProcessor()
file_exporter = FileExporter()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    # 按文件名排序（支持连续截图）
    sorted_files = sorted(files, key=lambda f: f.filename)
    
    for file in sorted_files:
        if file.filename == '':
            continue
        
        # 读取图片数据
        image_data = file.read()
        
        # OCR识别
        raw_text = ocr_processor.ocr_image(image_data)
        
        # AI文本处理（增强清理）
        # 先用开发者清洗命令去除多余换行，再做格式化
        cleaned_text = text_processor.dev_clean_text(raw_text)
        processed_text = text_processor.clean_and_format(cleaned_text)

        
        results.append({
            'filename': file.filename,
            'raw_text': raw_text,
            'processed_text': processed_text
        })
    
    # 检查是否有OCR失败的结果
    failed_results = [r for r in results if r['raw_text'].startswith(('无法连接OCR服务', 'API错误', 'OCR识别错误'))]
    if failed_results:
        # 返回部分成功的结果，即使有失败
        successful_results = [r for r in results if not r['raw_text'].startswith(('无法连接OCR服务', 'API错误', 'OCR识别错误'))]
        if successful_results:
            page_texts = [result['processed_text'] for result in successful_results]
            merged_text = text_processor.clean_and_merge_texts(page_texts)
            return jsonify({
                'results': results,
                'merged_text': merged_text,
                'warning': f'{len(failed_results)}张图片识别失败，已处理{len(successful_results)}张成功的图片'
            })
        else:
            return jsonify({
                'error': '所有图片OCR识别失败，请检查网络连接或配置API密钥',
                'results': results
            }), 400
    
    # 使用新的清理和合并函数
    page_texts = [result['processed_text'] for result in results]
    merged_text = text_processor.clean_and_merge_texts(page_texts)
    
    # 可选：进行语义修正
    if len(merged_text) > 1000:
        merged_text = text_processor.llm_semantic_correction(merged_text)
    
    # 返回合并后的结果
    return jsonify({
        'results': results,
        'merged_text': merged_text
    })
    
    return jsonify({'results': results})

@app.route('/dev/clean', methods=['POST'])
def dev_clean():
    """
    🧹 开发者接口：测试 OCR 文本的换行清理
    用法：
        POST { "text": "OCR原文" }
    返回：
        {"cleaned_text": "..."}
    """
    data = request.get_json()
    text = data.get('text', '')
    cleaned = text_processor.dev_clean_text(text)
    return jsonify({"cleaned_text": cleaned})




@app.route('/export', methods=['POST'])
def export_file():
    data = request.json
    text = data.get('text', '')
    format_type = data.get('format', 'txt')
    filename = data.get('filename', f'ocr_result_{uuid.uuid4().hex[:8]}')
    
    try:
        if format_type == 'txt':
            filepath = file_exporter.export_txt(text, filename)
        elif format_type == 'docx':
            filepath = file_exporter.export_docx(text, filename)
        elif format_type == 'md':
            filepath = file_exporter.export_md(text, filename)
        elif format_type == 'pdf':
            filepath = file_exporter.export_pdf(text, filename)
        else:
            return jsonify({'error': '不支持的文件格式'}), 400
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

# Vercel需要导出app对象
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5005)