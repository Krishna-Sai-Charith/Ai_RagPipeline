# 🤖 RAG-Based Chatbot with LangChain, Chroma, and Gradio

This project is a full-fledged implementation of a **Retrieval-Augmented Generation (RAG)** chatbot. It combines **semantic document search** with **interactive AI conversations** using LangChain, OpenAI's GPT-4o-mini, Chroma vector database, and Gradio UI.

You can upload markdown documents into a knowledge base, embed them as vectors, visualize their relationships in 2D/3D, and finally, interact with an intelligent chatbot that can answer questions based on your own content — not just pre-trained knowledge.

---

## 🚀 What This Project Does

- Loads Markdown documents from folders 📂  
- Splits them into chunks for better understanding ✂️  
- Generates **semantic vector embeddings** 🧠  
- Stores them in a Chroma vector database 💾  
- Visualizes embedding clusters in 2D and 3D 📈  
- Builds a conversational chain using LangChain 💬  
- Launches a browser-based chatbot using Gradio 🌐  

---

## 🧠 What Is RAG (Retrieval-Augmented Generation)?

RAG is a method where the AI model first **retrieves relevant knowledge** from your documents, then **generates a response** using that information.

flowchart LR
    Q[User Query] --> R[🔍 Retrieve Relevant Chunks]
    R --> L[🤖 LLM Generates Answer]

This approach:
- Keeps answers grounded in your data 🗂️  
- Avoids hallucinations from the LLM 🤯  
- Supports long-term knowledge base integration 🧩  

---

## 🔧 Key LangChain Abstractions

LangChain simplifies the RAG architecture with these main components:

| Component | Purpose | Emoji |
|----------|---------|--------|
| **LLM** | Generates natural language responses | 🧠 |
| **Retriever** | Finds relevant chunks from vector DB | 🔍 |
| **Memory** | Remembers previous conversation history | 💾 |

These are **assembled into a conversation chain**, where the user’s question triggers retrieval and response generation — all while maintaining chat history.

---

## 📦 Why We Need Vector Embeddings

Traditional keyword search fails to understand **semantic meaning**. For example, the words *car* and *vehicle* mean the same thing, but a keyword search wouldn’t know that.

### 🔢 Vector embeddings solve this by:
- Representing text as **high-dimensional numeric vectors**
- Capturing **context and meaning**, not just literal words
- Enabling **semantic similarity** search between chunks

### ✅ In this project:
- We used **OpenAI Embeddings** (`text-embedding-3-small`)
- Each document chunk was embedded into a vector
- These vectors were stored in **Chroma**, a fast vector DB
- When a user sends a message, similar vectors (chunks) are retrieved and passed to the LLM

---

## 🧩 How We Prepared the Documents

1. Markdown files were loaded from folders under `knowledge-base/`
2. LangChain split them into chunks of ~1000 characters with overlap
3. Each chunk was embedded into a vector
4. Vectors were stored persistently in `Chroma` for retrieval

This chunking allows:
- Better understanding of long documents
- Accurate chunk-level retrieval
- Efficient storage and search

---

## 📊 Embedding Visualization

To build intuition, we visualized the vector embeddings using **t-SNE**, a dimensionality reduction technique.

- 🔷 In **2D and 3D plots**, similar chunks appear close together
- 🔍 It shows how semantically similar documents cluster
- 📉 Useful for debugging embedding quality

This also helps explain how the **retriever** decides which chunks to fetch for a query.

---

## 💬 Conversational RAG with Memory

Using `ConversationalRetrievalChain`, we integrated:

- ✅ An **LLM** (GPT-4o-mini) to generate answers  
- ✅ A **Retriever** to fetch matching chunks from Chroma  
- ✅ A **Memory Buffer** to track conversation history  

This enables:
- Follow-up questions ("What about next year?")
- Context-aware answers
- Human-like, multi-turn conversations

---

## 🌐 Gradio Chat Interface

To make the system interactive, we wrapped everything in a **Gradio Chat Interface**.

- Type a question in your browser 🧑‍💻  
- The LLM retrieves and generates a response 💬  
- All past chat history is stored in memory 🧠  
- Runs locally using `inbrowser=True` for quick testing

---

## 🧪 Summary of the Flow

flowchart TD
    A[📂 Load Markdown Docs] --> B[✂️ Chunk with LangChain]
    B --> C[🔢 Create Embeddings with OpenAI]
    C --> D[💾 Store in Chroma Vector DB]
    D --> E[🔍 Use Retriever to Find Similar Chunks]
    E --> F[🤖 LLM Generates Answer with Context]
    F --> G[💬 Chat UI via Gradio]


---

## 🌟 Why This Project Matters

This project is a **real-world implementation** of modern LLM-based architecture used in:

- Enterprise knowledge bots  
- Custom GPTs and assistants  
- Search engines with semantic understanding  
- Internal document Q&A systems

You’ll walk away knowing:
- How to build your own RAG system  
- What embeddings actually represent  
- How memory enables context-aware chatbots  
- How to integrate everything into an interactive UI  

---

## 📁 Folder Structure

```
📂 knowledge-base/
├── 📄 products/
├── 📄 employees/
├── 📄 contracts/
└── 📄 company/
```

Place your `.md` documents into these folders to populate the knowledge base.

---

## ✅ Final Output

- A working chatbot powered by **your own documents**  
- A browser UI with memory-enabled conversations  
- Semantic document search and retrieval  
- 2D/3D embedding visualizations  

> It’s your **first step toward building your own ChatGPT over custom data!**
