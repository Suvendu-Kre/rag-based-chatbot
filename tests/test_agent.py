import pytest
from agents.main_agent import Agent

@pytest.fixture
def agent():
    return Agent()

def test_agent_responds_to_rag_design_query(agent):
    query = "Design a RAG pipeline for PDF documents."
    response = agent.run(query)
    assert "Architecture Diagram" in response
    assert "Step-by-step explanation" in response
    assert "Code snippets" in response
    assert "Design decisions" in response
    assert "Scaling considerations" in response
    assert "Possible improvements" in response

def test_agent_handles_legal_rag_query(agent):
    query = "Create a legal RAG system."
    response = agent.run(query)
    assert "Architecture Diagram" in response
    assert "Step-by-step explanation" in response

def test_agent_handles_optimization_query(agent):
    query = "Optimize RAG for low latency."
    response = agent.run(query)
    assert "batching embeddings" in response or "caching queries" in response or "async retrieval" in response

def test_agent_handles_faiss_qdrant_explanation_query(agent):
    query = "Explain FAISS vs Qdrant in this pipeline"
    response = agent.run(query)
    assert "FAISS" in response
    assert "Qdrant" in response