# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import pickle
import glob
import io
import base64
from docx import Document
from docx.oxml.ns import qn
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# -------- 段落处理工具 --------
def merge_segments(text, min_length=80):
    segments = [seg.strip() for seg in text.split('\n') if seg.strip()]
    merged_segments = []
    buffer = ""

    # 合并段落, 以确保每段至少有 min_length 个字符
    for seg in segments:
        buffer += seg
        if len(buffer) >= min_length:
            merged_segments.append(buffer)
            buffer = ""
        else:
            buffer += "\n"

    # 处理剩余的缓冲区, 如果缓冲区不为空且长度小于 min_length, 则将其附加到最后一个段落
    if buffer.strip():
        if len(buffer) >= min_length:
            merged_segments.append(buffer.strip())
        elif merged_segments:
            merged_segments[-1] += "\n" + buffer.strip()
        else:
            merged_segments.append(buffer.strip())
    return merged_segments or [text]

# -------- 文本清理 --------
def clean_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace(',', '，')
    text = re.sub(r'[\f\r]+', '', text)
    text = text.replace('\t', ' ')
    return text.strip()

# -------- 嵌入模型封装 --------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# -------- 核心 RAG 系统 --------
class RAGSystem:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_store_path = os.path.join(self.base_dir, "vector_store.faiss")
        self.documents_path = os.path.join(self.base_dir, "documents.pkl")
        self.pictures_dir = os.path.join(self.base_dir, "Pictures")
        os.makedirs(self.pictures_dir, exist_ok=True)
        self.embedder = SentenceTransformerEmbeddings(model_name)
        self.vector_store = None
        self.documents = []
        self.images = []
        self.load_vector_store()

    # 提取 DOCX 文档中的文本和图片
    def extract_text_from_docx(self, docx_path, min_paragraph_length=30):
        doc = Document(docx_path)
        full_text, images = [], []
        rels = doc.part.rels
        for para_idx, para in enumerate(doc.paragraphs):
            # 跳过标题样式的段落
            if para.style.name.startswith('Heading'):
                continue
            
            paragraph_text, para_images, image_counter = [], [], 0
            
            # 提取段落文本
            for run in para.runs:
                run_text = run.text
                if run._element.findall(qn('w:br')):
                    run_text = run_text.replace('\r', '\n')
                if run_text:
                    paragraph_text.append(run_text)

                # 提取图片
                for blip in run._element.findall('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}):
                    embed_id = blip.get(qn('r:embed'))
                    if embed_id and embed_id in rels:
                        image_counter += 1
                        position = len(full_text)
                        filename = f"paragraph_{position}_image_{image_counter}.png"
                        path = os.path.join(self.pictures_dir, filename)
                        try:
                            image = Image.open(io.BytesIO(rels[embed_id].target_part.blob))
                            image.save(path)
                            para_images.append({"filename": filename, "position": position, "rel_id": embed_id})
                        except Exception as e:
                            print(f"保存图片失败: {e}")
            # 合并段落文本
            if paragraph_text:
                combined = ''.join(paragraph_text).strip()
                # 丢弃过短的段落（可能是标题或无意义片段）
                if combined and len(combined) >= min_paragraph_length:
                    full_text.append(combined)
                    images.extend(para_images)

        # 文档级图片
        for rel_id, rel in rels.items():
            if "image" in rel.target_ref and rel_id not in [img["rel_id"] for img in images if "rel_id" in img]:
                filename = f"paragraph_unassigned_image_{len(images)+1}.png"
                path = os.path.join(self.pictures_dir, filename)
                try:
                    image = Image.open(io.BytesIO(rel.target_part.blob))
                    image.save(path)
                    images.append({"filename": filename, "position": -1, "rel_id": rel_id})
                except Exception as e:
                    print(f"保存图片失败: {e}")
        return full_text, images

    # 处理文件，调用提取文本和图片的方法
    def process_file(self, file_paths):
        all_text, all_images = [], []
        for file_path in file_paths:
            if not file_path.lower().endswith('.docx'):
                continue
            text_segments, images = self.extract_text_from_docx(file_path, min_paragraph_length=25)
            if text_segments:
                all_text.extend(text_segments)
                all_images.extend(images)
        if not all_text:
            raise ValueError("没有提取到任何文本")
        self.documents = all_text
        self.images = all_images
        return all_images

    # 加载向量存储
    def load_vector_store(self):
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, self.embedder, allow_dangerous_deserialization=True
                )
                if os.path.exists(self.documents_path):
                    with open(self.documents_path, 'rb') as f:
                        self.documents = pickle.load(f)
                print(f"向量存储已加载，共 {len(self.documents)} 条文档")
                return True
            except Exception as e:
                print(f"加载失败: {e}")
        return False

    # 保存向量存储
    def save_vector_store(self):
        if self.vector_store:
            self.vector_store.save_local(self.vector_store_path)
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print("向量存储和文档已保存")

    # 创建向量存储
    def create_vector_store(self):
        if not self.documents:
            raise ValueError("没有文档可用于嵌入")
        cleaned_segments, metadata = [], []
        for doc_idx, doc in enumerate(self.documents):
            cleaned = clean_text(doc)
            segments = merge_segments(cleaned)
            cleaned_segments.extend(segments)
            metadata.extend([{"original_doc_idx": doc_idx, "paragraph_number": doc_idx + 1}] * len(segments))
        self.vector_store = FAISS.from_texts(cleaned_segments, self.embedder, metadatas=metadata)
        self.documents = cleaned_segments
        print(f"创建向量存储，共 {len(cleaned_segments)} 段")
        self.save_vector_store()

    # 用户查询接口, 通过向量存储查询相关段落
    def query(self, question, k=4):
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        docs = self.vector_store.similarity_search(question, k=k)
        context, picture_path, seen_images = [], [], set()
        image_folder = os.path.join(self.base_dir, "Pictures")
        for doc in docs:
            paragraph_number = doc.metadata.get('paragraph_number', '未知')
            context.append(f"\n段落 {paragraph_number}: {doc.page_content}")
            if isinstance(paragraph_number, int):
                positions = [paragraph_number - 1, paragraph_number, paragraph_number + 1]
                for fname in os.listdir(image_folder):
                    for pos in positions:
                        if re.match(f"^paragraph_{pos}_image_\\d+\\.png$", fname) and fname not in seen_images:
                            picture_path.append(f"段落 {paragraph_number}: {os.path.join(image_folder, fname)}")
                            seen_images.add(fname)
        prompt = f"""参考以下《超声原理及生物医学工程应用：生物医学超声学》中的内容以及你的已有知识，对问题给出详细回答：{' '.join(context)} \n问题为: {question}"""
        return prompt, picture_path

# -------- 主入口 --------
def main():
    rag = RAGSystem()
    file_paths = glob.glob(os.path.join(rag.base_dir, "*.docx"))
    if not rag.vector_store:
        if not file_paths:
            raise FileNotFoundError("目录下没有 DOCX 文件")
        rag.process_file(file_paths)
        rag.create_vector_store()
    question = "超声换能器有哪些"
    prompt, images = rag.query(question)
    print(prompt)
    if images:
        print("\n相关图片：")
        for img in images:
            print(img)

if __name__ == "__main__":
    main()