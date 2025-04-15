import os
import re
import unicodedata
import pickle
import pypdf
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import glob
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

# 自定义嵌入类
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class RAGSystem:
    def __init__(self, model_name="BAAI/bge-large-zh"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_store_path = os.path.join(self.base_dir, "vector_store.faiss")
        self.documents_path = os.path.join(self.base_dir, "documents.pkl")

        self.embedder = SentenceTransformerEmbeddings(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len
        )
        self.vector_store = None
        self.documents = []
        self.load_vector_store()

    def clean_text(self, text):
        #  规范化Unicode字符（如全角转半角）
        text = unicodedata.normalize('NFKC', text)
        text = text.replace(',', '，')# 替换英文逗号为中文逗号
        text = re.sub(r'\n+', '', text)# 替换连续换行符为单个换行符
        text = re.sub(r'[\f\r]+', '', text)# 移除换页符和回车符
        text = text.replace('\t', ' ') # 替换制表符为单个空格
        text = re.sub(r'\s+', '', text)# 移除所有空格
        text = text.strip() #  去除首尾空格
        text = text.replace('\n', '') #  移除换行符
        return text

    def extract_text_from_pdf(self, pdf_path, use_ocr=False):
        text = ""
        if use_ocr:
            try:
                from surya.detection import DetectionPredictor
                from surya.recognition import RecognitionPredictor
                from PIL import Image

                # 初始化预测器
                if not hasattr(self, 'det_predictor'):
                    self.det_predictor = DetectionPredictor()
                    self.rec_predictor = RecognitionPredictor()

                # 将 PDF 转换为图像
                images = convert_from_path(pdf_path)
                all_text = ""
                for page_num, image in enumerate(images, 1):
                    try:
                        # 文本检测
                        det_predictions = self.det_predictor([image])
                        if not det_predictions or not det_predictions[0].bboxes:
                            print(f"页面 {page_num} 未检测到文本区域 ({pdf_path})")
                            continue

                        # 文本识别，指定语言为英文（可根据需要调整）
                        rec_predictions = self.rec_predictor([image], [["zh"]], self.det_predictor)
                        if rec_predictions and rec_predictions[0].text_lines:
                            for line in rec_predictions[0].text_lines:
                                all_text += line.text + " "
                        else:
                            print(f"页面 {page_num} 文本识别失败 ({pdf_path})")
                    except Exception as e:
                        print(f"页面 {page_num} OCR 处理失败 ({pdf_path}): {e}")
                        continue

                text = all_text.strip()
                if text:
                    print(f"已通过 Surya OCR 提取 PDF 文本: {pdf_path}")
                else:
                    print(f"通过 Surya OCR 提取 PDF 文本为空: {pdf_path}")
            except Exception as e:
                print(f"使用 Surya OCR 提取文本失败 ({pdf_path}): {e}")
                return ""
        else:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                        else:
                            print(f"页面 {page_num} 无可提取文本 ({pdf_path})")
                    print(f"已通过 pypdf 提取 PDF 文本: {pdf_path}")
            except Exception as e:
                print(f"PDF 提取失败 ({pdf_path}): {e}")
                return ""

        return self.clean_text(text)

    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                return self.clean_text(text)
        except Exception as e:
            print(f"TXT 提取失败 ({txt_path}): {e}")
            return ""

    def process_file(self, file_paths, use_ocr=False):
        all_text = ""
        for file_path in file_paths:
            abs_file_path = os.path.join(self.base_dir, file_path) if not os.path.isabs(file_path) else file_path
            if not os.path.exists(abs_file_path):
                print(f"文件 {abs_file_path} 不存在，跳过")
                continue

            if file_path.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(abs_file_path, use_ocr=use_ocr)
            elif file_path.lower().endswith('.txt'):
                text = self.extract_text_from_txt(abs_file_path)
            else:
                print(f"文件 {file_path} 格式不支持，仅支持 PDF 和 TXT，跳过")
                continue

            if text.strip():
                all_text += text + "\n"
            else:
                print(f"文件 {file_path} 提取的文本为空，跳过")

        if not all_text.strip():
            raise ValueError("所有文件提取的文本均为空")

        self.documents = self.text_splitter.split_text(all_text)
        print(f"已分割为 {len(self.documents)} 个文档块")

    def load_vector_store(self):
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embedder,
                    allow_dangerous_deserialization=True
                )
                print(f"已从 {self.vector_store_path} 加载向量存储")
                if os.path.exists(self.documents_path):
                    with open(self.documents_path, 'rb') as f:
                        self.documents = pickle.load(f)
                    print(f"已从 {self.documents_path} 加载文档")
                return True
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                self.vector_store = None
        return False

    def save_vector_store(self):
        if self.vector_store:
            try:
                self.vector_store.save_local(self.vector_store_path)
                print(f"向量存储已保存到 {self.vector_store_path}")
                with open(self.documents_path, 'wb') as f:
                    pickle.dump(self.documents, f)
                print(f"文档已保存到 {self.documents_path}")
            except Exception as e:
                print(f"保存向量存储失败: {e}")

    def create_vector_store(self):
        if self.vector_store:
            print("向量存储已存在，无需重新创建")
            return

        if not self.documents:
            raise ValueError("没有可嵌入的文档")

        self.vector_store = FAISS.from_texts(self.documents, self.embedder)
        print("向量存储已创建")
        self.save_vector_store()

    def query(self, question, k=3):
        if not self.vector_store:
            raise ValueError("向量存储尚未创建")

        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下内容回答问题：\n{context} \n问题: {question}"""
        print(prompt)
        return prompt


def main():
    rag = RAGSystem()
    file_paths = glob.glob(os.path.join(rag.base_dir, "*.pdf"))

    try:
        if not rag.vector_store:
            if not file_paths:
                raise FileNotFoundError("代码目录下未找到任何 PDF 文件")
            print(f"找到以下 PDF 文件: {file_paths}")
            rag.process_file(file_paths, use_ocr=True)
            if not rag.documents:
                print("pypdf 提取失败，尝试 Surya OCR...")
                rag.process_file(file_paths, use_ocr=True)
            rag.create_vector_store()

        question = "医学超声技术有哪些？"
        rag.query(question)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
