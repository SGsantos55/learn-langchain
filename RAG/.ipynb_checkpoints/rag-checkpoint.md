# Retrieval-Augmented Generation (RAG)

---

## 1. What RAG *Actually* Is 

**RAG = Information Retrieval + Text Generation**

LLMs **do not know private data** and **cannot search files**.
RAG forces the model to answer **using retrieved context** instead of guessing.

### Without RAG

```
Question → LLM → Answer (hallucination risk)
```

### With RAG

```
Question
 → Convert question to vector
 → Retrieve relevant chunks from your data
 → Inject chunks into prompt
 → LLM generates answer grounded in those chunks
```

LLM is **not trained again**. No fine-tuning here.

---

## 2. Core Components (High-Level Pipeline)

Every RAG system — regardless of framework — has these parts:

1. **Data Source** (PDFs, text, web, DB, notes)
2. **Loader** (convert source → text)
3. **Chunker** (split text)
4. **Embedding Model** (text → vectors)
5. **Vector Store** (store + search vectors)
6. **Retriever** (find relevant chunks)
7. **Prompt Template** (inject context)
8. **LLM** 

Mess up **any one** → bad RAG.

---

## 3. Document Object 

LangChain standardizes data as:

```python
Document(
    page_content="actual text",
    metadata={"source": "file.pdf", "page": 3}
)
```

Why metadata matters:

* Citations
* Debugging
* Filtering ("only this book", "only recent")

---

## 4. STEP 1 — Data Loading (NO embeddings yet)

### Rule

> Load **clean text first**. If this is wrong, stop.

All loaders output:

```python
List[Document]
```

---

### 4.1 Text Files

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("data.txt", encoding="utf-8")
docs = loader.load()
```

Use for:

* Notes
* Logs
* Plain datasets

---

### 4.2 Directory of Text Files

```python
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()
```

---

### 4.3 PDFs

#### PyPDFLoader (most common)

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("book.pdf")
docs = loader.load()
```

Each **page = one Document**.

#### PDFPlumberLoader (tables-heavy PDFs)

```python
from langchain.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("file.pdf")
docs = loader.load()
```

---

### 4.4 Web Pages (BeautifulSoup / bs4)

```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader([
    "https://example.com"
])

docs = loader.load()
```

Used for:

* Blogs
* Documentation sites

---

### 4.5 HTML Files

```python
from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("page.html")
docs = loader.load()
```

---

### 4.6 Markdown

```python
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md")
docs = loader.load()
```

---

### 4.7 CSV

```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader("data.csv")
docs = loader.load()
```

Each row → one Document.

---

### 4.8 JSON

```python
from langchain.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",
    text_content=False
)

docs = loader.load()
```

---

### Debug Always

```python
print(len(docs))
print(docs[0].page_content[:500])
print(docs[0].metadata)
```

---

## 5. STEP 2 — Chunking (MOST IMPORTANT STEP)

### Why chunking exists

* Embeddings work best on **focused meaning**
* Retrieval happens per chunk
* LLM context is limited

### Bad Chunking = Bad Retrieval

---

### Chunking Rules (Practical Defaults)

* Chunk size: **400–800 tokens** (start with 500)
* Overlap: **10–20%**
* Chunk by **characters**, not sentences

---

### LangChain Chunker

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
```

Each chunk is still a `Document`.

---

## 6. STEP 3 — Embeddings (Text → Vectors)

### What embeddings do

Convert meaning into numbers so we can **search by similarity**.

### Important

* Same embedding model must be used for:

  * documents
  * queries

---

### Example (HuggingFace)

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

## 7. STEP 4 — Vector Store

Stores vectors + metadata and allows fast similarity search.

### Common options

* FAISS (local, fast)
* Chroma (local, persistent)
* Pinecone / Milvus (cloud, scalable)

---

### FAISS Example

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    chunks,
    embedding=embeddings
)
```

---

## 8. STEP 5 — Retriever

Retriever wraps the vector DB.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

This returns **top-k relevant chunks**.

---

## 9. STEP 6 — Prompt Construction

LLMs must be **forced** to use retrieved context.

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
"""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}
"""
)
```

---

## 10. STEP 7 — LLM (Groq)



```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-70b-8192",
    api_key="YOUR_GROQ_API_KEY"
)
```

---

## 11. STEP 8 — Full RAG Chain

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

qa.run("Your question here")
```

---

## 12. Common Failure Modes (Read This Twice)

### ❌ Hallucinations

* Weak retrieval
* Bad chunking
* Prompt not strict

### ❌ Wrong Answers

* Irrelevant chunks
* Too many chunks

### ❌ Slow RAG

* Too many embeddings
* No caching

---

## 13. RAG ≠ Fine-Tuning

| RAG                | Fine-Tuning           |
| ------------------ | --------------------- |
| Uses external data | Changes model weights |
| Cheap              | Expensive             |
| Dynamic            | Static                |
| Easy to update     | Hard to update        |

---

## 14. Mental Checklist Before Production

* [ ] Loader output clean?
* [ ] Chunk size reasonable?
* [ ] Embedding model consistent?
* [ ] Retriever returns relevant chunks?
* [ ] Prompt forbids guessing?

---

## Final Truth

> **RAG is 80% retrieval quality, 20% LLM.**

Groq speed means nothing if retrieval is trash.

