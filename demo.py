from RAG.rag_system import RAGSystem

def call_rag_query(question, model_name="BAAI/bge-large-zh-v1.5"):
    rag = RAGSystem(model_name=model_name)  # 初始化 RAGSystem
    try:
        result, picture_path = rag.query(question, k = 4) # k 是返回的相似文本块数量，k=1 就是只输出最相关的一段文字
        return result, picture_path
    except Exception as e:
        print(f"查询失败: {e}")
        return None, []

question = "连续波多普勒超声换能器有哪几种形式？"  # 可连接到用户接口
result, picture_path = call_rag_query(question)

print(result)
print("\n关联图片路径：")
for path_info in picture_path:
    print(path_info)
