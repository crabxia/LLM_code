import os

# 强制清除系统代理配置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["all_proxy"] = ""


import subprocess
from langchain.chains import RetrievalQA #用于在检索增强生成（RAG）任务中结合检索（retrieval）模块和问答（question-answering）模型。支持基于用户问题从文档中提取相关信息并生成回答
from langchain_community.vectorstores import FAISS #用于管理嵌入向量的索引和快速检索
from langchain_huggingface import HuggingFaceEmbeddings #使用 Hugging Face 提供的模型（如 BERT、Sentence Transformers）生成嵌入向量，后续用于搜索、检索和分类。
from langchain.docstore.document import Document #用于表示文档对象，用于存储原始数据和关联元信息，供链式处理和检索任务使用。
from sentence_transformers import SentenceTransformer #SentenceTransformer 提供了简单的 API 来加载预训练模型，生成高质量的嵌入向量。嵌入向量可以用于搜索、聚类、分类等任务。
import gradio as gr #交互式 Web 应用。

# -------- 1. 调用 Llama 模型的函数 --------
def ollama_run(prompt, model="llama3.1"):
    """
    调用 Ollama 的 'run' 模式，运行指定模型。
    :param prompt: 用户输入的提示
    :param model: 使用的模型名称
    :return: 模型生成的回复
    """
    try:
        # 构建命令行调用
        command = ["ollama", "run", model]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 输入提示并获取输出
        stdout, stderr = process.communicate(input=prompt)
        
        if process.returncode == 0:
            return stdout.strip()
        else:
            raise Exception(f"调用失败: {stderr.strip()}")
    except Exception as e:
        raise Exception(f"运行 Ollama 出现错误: {str(e)}")

# -------- 2. 构建向量数据库 --------
def build_vectorstore():
    """
    创建一个简单的向量数据库，包含演示文档。
    """
    sentences = [
        "Llama 3.1 是最新版本，支持更复杂的自然语言处理任务。",
        "与之前版本相比，它在推理和生成速度上有显著提升。",
        "Llama 模型非常适合生成式 AI 应用，包括问答和内容创作。"
    ] 
    
    # 使用 HuggingFaceEmbeddings 来处理 SentenceTransformer
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # 使用 FAISS 从文本和嵌入创建向量数据库
    vectorstore = FAISS.from_texts(sentences, embedding_model)
    
    return vectorstore

vectorstore = build_vectorstore()

# -------- 3. RAG 系统 --------
def rag_with_ollama(query):
    """
    RAG 系统，通过检索相关文档构建上下文，并调用 Ollama 模型生成回答。
    """
    # 检索相关文档
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 构建提示词
    prompt = f"以下是相关背景信息：\n{context}\n\n基于以上信息，回答问题：{query}"
    return ollama_run(prompt)

# -------- 4. Gradio 用户界面 --------
def chat_with_bot(user_input):
    """
    处理用户输入，调用 RAG 系统生成回复。
    """
    try:
        response = rag_with_ollama(user_input)
        return response
    except Exception as e:
        return f"出现错误: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("### 基于 Llama 3.1 的个性化聊天机器人")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="输入您的问题", placeholder="请输入您的问题...")
        with gr.Column():
            bot_response = gr.Textbox(label="机器人回复", interactive=False)

    submit = gr.Button("发送")
    submit.click(chat_with_bot, inputs=user_input, outputs=bot_response)

# 启动界面
print("准备启动 Gradio 界面...")
demo.launch()
