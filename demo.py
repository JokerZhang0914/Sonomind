from rag_system import RAGSystem  

def call_rag_query(question, model_name="BAAI/bge-large-zh"):
    rag = RAGSystem(model_name=model_name) # 初始化 RAGSystem
    if not rag.vector_store: # 检查是否已加载向量存储
        raise ValueError("向量存储不存在，请确保已创建并保存向量存储")
    try: # 调用 query 方法
        result = rag.query(question, k=3) # k 是返回的相似文本块数量，k=1就是只输出最相关的一段文字
        return result
    except Exception as e:
        print(f"查询失败: {e}")
        return None

question = "基于相控聚焦的多元线性阵列探测系统是什么？" # 可连接到用户接口
result = call_rag_query(question)   # 查询后获得的prompt
if result:
    print("查询结果：", result)