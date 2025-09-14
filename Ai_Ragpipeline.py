# =============================================
# 🤖 RAG Pipeline with Visualization & Chatbot
# ✅ Runs in VS Code
# =============================================

import os
import glob
from dotenv import load_dotenv
import gradio as gr

# LangChain & related
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Plotting
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# ---------------------------------------------
# ⚙️ CONFIG
# ---------------------------------------------
MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# ---------------------------------------------
# 📄 Load Documents
# ---------------------------------------------
print("🔄 Loading documents...")

folders = glob.glob("knowledge-base/*")
text_loader_kwargs = {'encoding': 'utf-8'}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"✅ Loaded {len(documents)} documents.")

# ---------------------------------------------
# ✂️ Split into Chunks
# ---------------------------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"🧩 Split into {len(chunks)} chunks.")

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"📂 Document types found: {', '.join(doc_types)}")

# ---------------------------------------------
# 🔢 Embeddings + Chroma Vector DB
# ---------------------------------------------
embeddings = OpenAIEmbeddings()

# Clean DB if exists
if os.path.exists(db_name):
    print("🧹 Cleaning up existing Chroma DB...")
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create new Vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"📦 Vectorstore created with {vectorstore._collection.count()} chunks.")

# Check vector dimension
sample_embedding = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0]
print(f"📐 Vector dimension: {len(sample_embedding)}")

# ---------------------------------------------
# 📉 t-SNE Visualization
# ---------------------------------------------
print("📊 Preparing t-SNE visualization...")

result = vectorstore._collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
color_map = {'products': 'blue', 'employees': 'green', 'contracts': 'red', 'company': 'orange'}
colors = [color_map.get(t, 'gray') for t in doc_types]

# 2D t-SNE
tsne_2d = TSNE(n_components=2, random_state=42)
reduced_2d = tsne_2d.fit_transform(vectors)

fig_2d = go.Figure(data=[go.Scatter(
    x=reduced_2d[:, 0],
    y=reduced_2d[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])
fig_2d.update_layout(
    title='2D Chroma Vector Store Visualization',
    xaxis_title='x', yaxis_title='y',
    width=800, height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig_2d.show()

# 3D t-SNE
tsne_3d = TSNE(n_components=3, random_state=42)
reduced_3d = tsne_3d.fit_transform(vectors)

fig_3d = go.Figure(data=[go.Scatter3d(
    x=reduced_3d[:, 0],
    y=reduced_3d[:, 1],
    z=reduced_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])
fig_3d.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900, height=700,
    margin=dict(r=20, b=10, l=10, t=40)
)
fig_3d.show()

# ---------------------------------------------
# 💬 RAG Chatbot Setup
# ---------------------------------------------
print("🧠 Initializing RAG Chatbot...")

llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Sample query
query = "Can you describe Insurellm in a few sentences?"
response = conversation_chain.invoke({"question": query})
print("🗣️ Bot Response:", response["answer"])

# ---------------------------------------------
# 💻 Gradio Chat Interface
# ---------------------------------------------
print("🚀 Launching Gradio Chat Interface...")

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

gr.ChatInterface(chat, type="messages").launch(inbrowser=True)
