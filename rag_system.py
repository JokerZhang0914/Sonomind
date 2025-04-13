import os
import re
import pickle
import pypdf
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
import glob

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
        # 获取代码所在目录
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
        """
        清理文本，去除多余空格、换行、特殊字符和全角字符
        """
        fullwidth_map = str.maketrans(
            '０１２３４５６７８９ＭＨｚ',
            '0123456789MHz'
        )
        text = text.translate(fullwidth_map)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(c for c in text if c.isprintable() and (c.isalnum() or c.isspace() or c in '.,;:-()[]{}<>?!'))
        return text.strip()

    def preprocess_image_for_ocr(self, image):
        """
        预处理图像以提高OCR准确性
        """
        image = image.convert('L')
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        img_array = np.array(image)
        threshold = 128
        img_array = (img_array > threshold) * 255
        return Image.fromarray(img_array.astype(np.uint8))

    def extract_text_from_pdf(self, pdf_path, use_ocr=False):
        text = ""
        if use_ocr:
            try:
                if not pytesseract.get_tesseract_version():
                    raise RuntimeError("Tesseract 未安装或未正确配置")
                poppler_path = r"C:\Program Files\poppler\bin"  # 替换为你的 Poppler bin 路径
                images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
                for i, image in enumerate(images):
                    image = self.preprocess_image_for_ocr(image)
                    page_text = pytesseract.image_to_string(image, lang='chi_sim', config='--psm 6')
                    text += page_text + "\n"
                    print(f"已通过OCR提取第 {i+1} 页 ({pdf_path})")
                print(f"OCR 提取完成: {pdf_path}")
            except Exception as e:
                print(f"OCR提取失败 ({pdf_path}): {e}")
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
                    print(f"已通过pypdf提取PDF文本: {pdf_path}")
            except Exception as e:
                print(f"PDF提取失败 ({pdf_path}): {e}")
                return ""
        
        return self.clean_text(text)

    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                return self.clean_text(text)
        except Exception as e:
            print(f"TXT提取失败 ({txt_path}): {e}")
            return ""

    def process_file(self, file_paths, use_ocr=False):
        """
        处理多个文件，合并文本后分块
        """
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
                print(f"文件 {file_path} 格式不支持，仅支持PDF和TXT，跳过")
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
        """
        尝试加载代码目录下的向量存储和文档
        """
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
        """
        保存向量存储和文档到代码目录
        """
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
        """
        创建向量存储，如果不存在则创建并保存
        """
        if self.vector_store:
            print("向量存储已存在，无需重新创建")
            return
        
        if not self.documents:
            raise ValueError("没有可嵌入的文档")
        
        self.vector_store = FAISS.from_texts(self.documents, self.embedder)
        print("向量存储已创建")
        self.save_vector_store()

    def query(self, question, k=5):
        if not self.vector_store:
            raise ValueError("向量存储尚未创建")
        
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下内容回答问题：
{context}
问题: {question}"""
        print(prompt)
        return prompt

def main():
    rag = RAGSystem()
    # 使用 glob 查找代码目录下的所有 PDF 文件
    file_paths = glob.glob(os.path.join(rag.base_dir, "*.pdf"))
    
    try:
        if not rag.vector_store:
            if not file_paths:
                raise FileNotFoundError("代码目录下未找到任何 PDF 文件")
            print(f"找到以下 PDF 文件: {file_paths}")
            rag.process_file(file_paths, use_ocr=False)
            if not rag.documents:
                print("pypdf提取失败，尝试OCR...")
                rag.process_file(file_paths, use_ocr=True)
            rag.create_vector_store()
        
        question = "基于相控聚焦的多元线性阵列探测系统是什么？"
        rag.query(question)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()