import os
import re
import unicodedata
import pickle
from docx import Document
from docx.oxml.ns import qn
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from PIL import Image
import numpy as np
import glob
import io
import base64

# 自定义嵌入类
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_documents(self, texts):
        all_segments = []
        for text in texts:
            segments = [seg.strip() for seg in text.split('\n') if seg.strip()]
            merged_segments = []
            buffer = ""
            
            for seg in segments:
                buffer += seg
                if len(buffer) >= 50:
                    merged_segments.append(buffer)
                    buffer = ""
                else:
                    buffer += "\n"
            
            if buffer.strip() and len(buffer) >= 50:
                merged_segments.append(buffer.strip())
            elif buffer.strip() and merged_segments:
                merged_segments[-1] += "\n" + buffer.strip()
            
            all_segments.extend(merged_segments if merged_segments else [text])
        
        embeddings = self.model.encode(all_segments, show_progress_bar=True).tolist()
        return embeddings

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# RAG系统类
class RAGSystem:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vector_store_path = os.path.join(self.base_dir, "vector_store.faiss")
        self.documents_path = os.path.join(self.base_dir, "documents.pkl")
        self.pictures_dir = os.path.join(self.base_dir, "Pictures")
        if not os.path.exists(self.pictures_dir):
            os.makedirs(self.pictures_dir)
        self.embedder = SentenceTransformerEmbeddings(model_name)
        
        self.vector_store = None
        self.documents = []
        self.images = []
        self.load_vector_store()

    def clean_text(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = text.replace(',', '，')
        text = re.sub(r'[\f\r]+', '', text)
        text = text.replace('\t', ' ')
        text = text.strip()
        return text

    def extract_text_from_docx(self, docx_path):
        doc = Document(docx_path)
        full_text = []
        images = []
        rels = doc.part.rels

        # 遍历段落
        for para_idx, para in enumerate(doc.paragraphs):
            paragraph_text = []
            para_images = []
            image_counter = 0
            
            for run in para.runs:
                run_text = run.text
                br_elements = run._element.findall(qn('w:br'))
                if br_elements:
                    run_text = run_text.replace('\r', '\n')
                if run_text:
                    paragraph_text.append(run_text)
                
                # 检查嵌入图片
                for blip in run._element.findall('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}):
                    embed_id = blip.get(qn('r:embed'))
                    if embed_id and embed_id in rels:
                        image_counter += 1
                        position = len(full_text)
                        image_filename = f"paragraph_{position}_image_{image_counter}.png"
                        image_path = os.path.join(self.pictures_dir, image_filename)
                        image_data = rels[embed_id].target_part.blob
                        try:
                            image = Image.open(io.BytesIO(image_data))
                            image.save(image_path)
                            para_images.append({"filename": image_filename, "position": position, "rel_id": embed_id})
                            print(f"图片 {image_filename} 分配到段落 {position} (para_idx={para_idx})")
                        except Exception as e:
                            print(f"保存图片 {image_filename} 失败: {e}")
            
            # 仅当段落有有效文本时添加到 full_text
            if paragraph_text:
                combined_text = ''.join(paragraph_text).strip()
                if combined_text:
                    full_text.append(combined_text)
                    images.extend(para_images)

        # 处理未分配的文档级图片
        doc_image_counter = 0
        for rel_id, rel in rels.items():
            if "image" in rel.target_ref and rel_id not in [img["rel_id"] for img in images if "rel_id" in img]:
                doc_image_counter += 1
                image_filename = f"paragraph_unassigned_image_{doc_image_counter}.png"
                image_path = os.path.join(self.pictures_dir, image_filename)
                image_data = rel.target_part.blob
                try:
                    image = Image.open(io.BytesIO(image_data))
                    image.save(image_path)
                    images.append({"filename": image_filename, "position": -1, "rel_id": rel_id})
                    print(f"图片 {image_filename} 未分配段落（position: -1）")
                except Exception as e:
                    print(f"保存图片 {image_filename} 失败: {e}")
        
        print(f"提取 {len(full_text)} 个文档块，{len(images)} 张图片")
        return full_text, images

    def process_file(self, file_paths):
        all_text_segments = []
        all_images = []
        for file_path in file_paths:
            abs_file_path = os.path.join(self.base_dir, file_path) if not os.path.isabs(file_path) else file_path 
            if file_path.lower().endswith('.docx'):
                text_segments, images = self.extract_text_from_docx(abs_file_path)
                all_images.extend(images)
                all_text_segments.extend(text_segments)
            else:
                print(f"文件 {file_path} 格式不支持，仅支持 docx跳过")
                continue

            if text_segments:
                if images:
                    image_filenames = [img["filename"] for img in images]
                    print(f"从 {file_path} 提取到 {len(images)} 张图片")
                    print(f"相关图像：{ '、'.join(image_filenames) }")
                    print(f"图片段落分配：{[f'{img['filename']} -> 段落 {img['position']}' for img in images]}")
            else:
                print(f"文件 {file_path} 提取的文本为空，跳过")

        if not all_text_segments:
            raise ValueError("所有文件提取的文本均为空")

        self.documents = all_text_segments
        self.images = all_images
        print(f"总计分割为 {len(self.documents)} 个文档块")
        return all_images

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

        cleaned_segments = []
        metadata = []
        for doc_idx, doc in enumerate(self.documents):
            cleaned_doc = self.clean_text(doc)
            segments = [seg.strip() for seg in cleaned_doc.split('\n') if seg.strip()]
            merged_segments = []
            buffer = ""
            
            for seg in segments:
                buffer += seg
                if len(buffer) >= 50:
                    merged_segments.append(buffer)
                    buffer = ""
                else:
                    buffer += "\n"
            
            if buffer.strip() and len(buffer) >= 50:
                merged_segments.append(buffer.strip())
            elif buffer.strip() and merged_segments:
                merged_segments[-1] += "\n" + buffer.strip()
            
            cleaned_segments.extend(merged_segments if merged_segments else [cleaned_doc])
            metadata.extend([{"original_doc_idx": doc_idx, "paragraph_number": doc_idx + 1}] * len(merged_segments if merged_segments else [cleaned_doc]))

        if not cleaned_segments:
            raise ValueError("清理后没有有效的文档")

        self.vector_store = FAISS.from_texts(cleaned_segments, self.embedder, metadatas=metadata)
        self.documents = cleaned_segments
        print(f"向量存储已创建，包含 {len(cleaned_segments)} 个子段落")
        self.save_vector_store()

    def query(self, question, k=1):
        picture_path = []  # 使用列表存储图片路径
        if not self.vector_store:
            raise ValueError("向量存储尚未创建")

        docs = self.vector_store.similarity_search(question, k=k)
        context = []
        image_folder = "Pictures"  # 假设所有图片保存在 pictures 文件夹中

        for doc in docs:
            paragraph_number = doc.metadata.get('paragraph_number', '未知')
            content = f"段落 {paragraph_number}: {doc.page_content}"
            context.append(content)

            # 检查当前段落及其前后2个位置是否有图片
            if isinstance(paragraph_number, int):  # 确保段落编号是整数
                target_positions = [paragraph_number - 1, paragraph_number, paragraph_number + 1]
                related_images = []

                if os.path.exists(image_folder):
                    for fname in os.listdir(image_folder):
                        for pos in target_positions:
                            pattern = f"^paragraph_{pos}_image_\\d+\\.png$"
                            if re.match(pattern, fname):
                                related_images.append(os.path.join(image_folder, fname))

                if related_images:
                    picture_path.append(f"段落 {paragraph_number}: {', '.join(related_images)}")

        prompt = f"""参考以下内容以及你的已有知识回答问题：\n{'\n'.join(context)} \n问题为: {question}"""       
        return prompt, picture_path


def main():
    rag = RAGSystem()
    file_paths = glob.glob(os.path.join(rag.base_dir, "*.docx"))

    try:
        if not rag.vector_store:
            if not file_paths:
                raise FileNotFoundError("代码目录下未找到任何 DOCX 文件")
            print(f"找到以下 DOCX 文件: {file_paths}")
            images = rag.process_file(file_paths)
            if not rag.documents:
                raise ValueError("文档提取失败")
            rag.create_vector_store()
            if images:
                image_filenames = [img["filename"] for img in images]
                print(f"共提取到 {len(images)} 张图片")
                print(f"相关图像：{ '、'.join(image_filenames) }")

        question = "连续波多普勒超声换能器有哪几种形式"
        rag.query(question)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()