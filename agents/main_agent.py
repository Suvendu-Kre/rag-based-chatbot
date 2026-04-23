from langchain_google_vertexai import ChatVertexAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
import os
from typing import List

class Agent:
    def __init__(self):
        self.llm = ChatVertexAI(model_name="gemini-2.0-flash-001", project="gen-ai-poc-onboarding", location="us-central1")
        self.tools_map = {t.name: t for t in self._get_tools()}
        self.llm_with_tools = self.llm.bind_tools(list(self.tools_map.values()))
        self.system_prompt = """You are an expert AI system called **RAG Architect Agent** specialized in designing and generating production-ready Retrieval-Augmented Generation (RAG) pipelines.

## 🎯 Objective

Your task is to design, explain, and generate complete RAG-based chatbot systems using the following mandatory components:

* Sentence-based chunking (semantic-aware splitting preferred)
* BGE-M3 embeddings (multi-lingual, dense + sparse support)
* FAISS index (for fast in-memory similarity search)
* Qdrant DB (as the primary persistent vector database)

---

## 🧠 Core Responsibilities

When given any input (documents, URLs, raw text, or use-case description), you MUST:

### 1. Understand the Use Case

* Identify domain (e.g., legal, chatbot, docs QA, support bot)
* Determine scale (small / medium / large dataset)
* Identify latency vs accuracy tradeoffs

---

### 2. Design the RAG Pipeline (MANDATORY STRUCTURE)

You MUST always structure the pipeline as:

1. Data Ingestion
2. Text Preprocessing
3. Sentence Chunking
4. Embedding Generation (BGE-M3)
5. Dual Indexing:

   * FAISS (local fast retrieval)
   * Qdrant (persistent + scalable retrieval)
6. Query Pipeline:

   * Query embedding
   * Hybrid retrieval (FAISS + Qdrant or fallback strategy)
7. Context Re-ranking (optional but recommended)
8. LLM Generation Layer
9. Response Formatting

---

### 3. Chunking Rules (STRICT)

* Use sentence-level splitting (not naive token splitting)
* Maintain semantic coherence
* Recommended:

  * Chunk size: 200–500 tokens
  * Overlap: 20–50 tokens

Explain WHY you chose these values.

---

### 4. Embedding Rules

* Use **BGE-M3**
* Mention:

  * dense vectors
  * sparse vectors (if hybrid search used)
* Normalize embeddings if needed
* Explain embedding dimensionality and impact

---

### 5. Indexing Strategy (VERY IMPORTANT)

You MUST explain:

#### FAISS:

* Type (Flat, IVF, HNSW)
* Why chosen (speed vs accuracy)

#### Qdrant:

* Collection schema
* Vector config
* Payload structure
* Filtering capability

Explain how FAISS and Qdrant complement each other.

---

### 6. Retrieval Strategy

You MUST include one of:

* Hybrid retrieval (dense + sparse)
* OR dual-stage retrieval:

  * FAISS (fast shortlist)
  * Qdrant (refined search)

Explain trade-offs clearly.

---

### 7. Code Generation (MANDATORY)

Always generate:

* Python code using:

  * sentence-transformers or HuggingFace (for BGE-M3)
  * faiss
  * qdrant-client
* Modular structure:

  * ingestion.py
  * embedding.py
  * index.py
  * query.py

---

### 8. Output Format (STRICT)

Your response MUST contain:

## ✅ Architecture Diagram (text or mermaid)

## ✅ Step-by-step explanation

## ✅ Code snippets

## ✅ Design decisions

## ✅ Scaling considerations

## ✅ Possible improvements

---

### 9. Optimization Guidelines

Include:

* batching embeddings
* caching queries
* async retrieval
* GPU vs CPU tradeoffs

---

### 10. Tone & Style

* Clear, structured, developer-friendly
* Avoid unnecessary theory
* Focus on practical implementation

---

## 🚫 Constraints

* DO NOT skip FAISS or Qdrant
* DO NOT use naive chunking
* DO NOT give vague answers
* ALWAYS justify design decisions

---

## 🧪 Example Inputs You Should Handle

* "Build a chatbot for PDF documents"
* "Create a legal RAG system"
* "Optimize RAG for low latency"
* "Explain FAISS vs Qdrant in this pipeline"

---

## 🏁 Goal

Produce a **production-ready, scalable, and optimized RAG system design + implementation** every time.
"""
        self._faiss_index = None
        self._faiss_docs: list = []
        self._embedder = None

    def _get_tools(self):
        from tools.tool_manager import get_tools
        return get_tools()

    def run(self, message: str) -> str:
        retrieved_chunks = self._retrieve_from_rag(message)
        context = "\n".join(retrieved_chunks)
        messages = [
            SystemMessage(content=self.system_prompt + f"\n\nContext from RAG:\n{context}"),
            HumanMessage(content=message),
        ]
        # Agentic loop: keep calling LLM until no more tool calls
        for _ in range(3):  # max iterations to prevent infinite loops
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            # Execute each tool call and feed results back
            for tc in response.tool_calls:
                fn = self.tools_map.get(tc["name"])
                tool_result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
                self._ingest_to_rag(message, str(tool_result), tc["name"])

        return response.content if hasattr(response, "content") else str(response)

    def _ingest_to_rag(self, query: str, mcp_result: str, tool_name: str = 'mcp_tool') -> None:
        """Ingest MCP-fetched data using BGE-M3, FAISS, Qdrant as specified in the system prompt."""
        try:
            import os
            from datetime import datetime
            text = f'Tool: {tool_name}\nQuery: {query}\nResult: {mcp_result}'
                # Sentence-level chunking: split text into sentences, group into ~300-token chunks
            import re as _re
            sentences = _re.split(r'(?<=[.!?])\s+', text)
            chunk_size, overlap, chunks, current = 300, 50, [], []
            for s in sentences:
                current.append(s)
                if sum(len(x.split()) for x in current) >= chunk_size:
                    chunks.append(' '.join(current))
                    current = current[-overlap:] if overlap else []
            if current:
                chunks.append(' '.join(current))
            from sentence_transformers import SentenceTransformer
            if self._embedder is None:
                self._embedder = SentenceTransformer('BAAI/bge-m3')
            embeddings = self._embedder.encode(chunks, normalize_embeddings=True).tolist()
            # FAISS: build/update in-memory index for fast local retrieval
            import faiss, numpy as np
            vecs = np.array(embeddings, dtype='float32')
            if not hasattr(self, '_faiss_index') or self._faiss_index is None:
                self._faiss_index = faiss.IndexFlatIP(vecs.shape[1])  # inner-product (cosine after normalise)
                self._faiss_docs: list = []
            self._faiss_index.add(vecs)
            self._faiss_docs.extend(chunks)
            # Qdrant: upsert into persistent vector DB
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            import hashlib as _hl
            qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
            qdrant_key = os.environ.get('QDRANT_API_KEY', '')
            qc = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
            _COL = 'knowledge_base'
            if _COL not in [c.name for c in qc.get_collections().collections]:
                qc.create_collection(_COL, vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE))
            points = [
                PointStruct(
                    id=int(_hl.md5(f'{tool_name}:{query}:{i}'.encode()).hexdigest()[:8], 16),
                    vector=embeddings[i],
                    payload={'text': chunk, 'tool': tool_name, 'query': query}
                )
                for i, chunk in enumerate(chunks)
            ]
            qc.upsert(collection_name=_COL, points=points)
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f'RAG ingestion failed: {e}')

    def _retrieve_from_rag(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks from FAISS and Qdrant."""
        try:
            import faiss
            import numpy as np
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, Range
            import os
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('BAAI/bge-m3')

            query_embedding = self._embedder.encode(query, normalize_embeddings=True)
            # FAISS retrieval (fast shortlist)
            if self._faiss_index is None:
                return []

            k_faiss = min(2 * top_k, self._faiss_index.ntotal)  # avoid out-of-bounds
            distances, indices = self._faiss_index.search(np.array([query_embedding], dtype='float32'), k_faiss)
            faiss_results = [(self._faiss_docs[i], distances[0][idx]) for idx, i in enumerate(indices[0]) if i < len(self._faiss_docs)]

            # Qdrant retrieval (persistent search)
            qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
            qdrant_key = os.environ.get('QDRANT_API_KEY', '')
            qc = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
            _COL = 'knowledge_base'
            qdrant_results = qc.search(
                collection_name=_COL,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=None  # Add filters if needed
            )
            # Combine and re-rank (simple concatenation for now)
            combined_results = [(r.payload['text'], r.score) for r in qdrant_results]
            combined_results.extend([(text, -distance) for text, distance in faiss_results]) # Use negative distance as score

            # Sort by score and return top_k unique chunks
            unique_chunks = []
            seen = set()
            for chunk, score in sorted(combined_results, key=lambda x: x[1], reverse=True):
                if chunk not in seen:
                    unique_chunks.append(chunk)
                    seen.add(chunk)
                    if len(unique_chunks) >= top_k:
                        break
            return unique_chunks

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f'RAG retrieval failed: {e}')
            return []