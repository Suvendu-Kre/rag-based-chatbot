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
        self.system_prompt = """You are a helpful AI agent. Your primary function is to assist users by leveraging a knowledge base to provide accurate and contextually relevant answers. You combine retrieval-augmented generation with conversational AI for enhanced knowledge-driven interactions. You use BGE-M3 embeddings, FAISS index, and Qdrant vector DB for RAG. You use sentence-level chunking (200-500 tokens, 20-50 overlap). You use FAISS for fast shortlist and Qdrant for persistent storage."""
        self._faiss_index = None
        self._faiss_docs: list = []
        self._embedder = None

    def _get_tools(self):
        from tools.tool_manager import get_tools
        return get_tools()

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
            import faiss, numpy as np
            from sentence_transformers import SentenceTransformer
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, Range
            import os
            if self._embedder is None:
                self._embedder = SentenceTransformer('BAAI/bge-m3')

            query_embedding = self._embedder.encode(query, normalize_embeddings=True)
            # FAISS search (fast shortlist)
            if self._faiss_index is None:
                return []
            distances, indices = self._faiss_index.search(np.array([query_embedding], dtype='float32'), k=2 * top_k)
            candidate_chunks = [self._faiss_docs[i] for i in indices[0]]

            # Qdrant search (persistent confirmation)
            qdrant_url = os.environ.get('QDRANT_URL', 'http://localhost:6333')
            qdrant_key = os.environ.get('QDRANT_API_KEY', '')
            qc = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)
            _COL = 'knowledge_base'
            search_result = qc.search(
                collection_name=_COL,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=Filter(
                    must=[FieldCondition(key="text", range=Range(gte=1))]
                )
            )
            qdrant_chunks = [hit.payload['text'] for hit in search_result]

            # Combine and deduplicate results, prioritizing Qdrant
            retrieved_chunks = list(dict.fromkeys(qdrant_chunks + candidate_chunks))  # Preserve order

            return retrieved_chunks[:top_k]  # Return top_k

        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f'RAG retrieval failed: {e}')
            return []

    def run(self, message: str) -> str:
        # Retrieve context from RAG
        context = self._retrieve_from_rag(message)
        context_str = "\n".join(context)

        messages = [
            SystemMessage(content=self.system_prompt + f"\n\nRelevant context:\n{context_str}"),
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
                self._ingest_to_rag(message, str(tool_result), tc["name"]) # Ingest data after tool call
        return response.content if hasattr(response, "content") else str(response)

    async def chat(self, message: str) -> str:
        return self.run(message)