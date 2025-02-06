import os

# 强制清除系统代理配置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["all_proxy"] = ""


import subprocess
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import gradio as gr

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

int main
    ollama_run(prompt)
