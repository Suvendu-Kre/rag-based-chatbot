import pytest
from agents.main_agent import Agent

@pytest.fixture
def agent():
    return Agent()

def test_agent_initialization(agent):
    assert agent is not None
    assert agent.system_prompt is not None

def test_agent_rag_design(agent):
    # A simple test to see if the agent provides a RAG design
    response = agent.run("Design a RAG pipeline for document retrieval.")
    assert "Architecture Diagram" in response
    assert "Step-by-step explanation" in response