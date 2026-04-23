import pytest
from agents.main_agent import Agent

@pytest.fixture
def agent():
    return Agent()

def test_agent_initialization(agent):
    assert agent is not None
    assert agent.system_prompt is not None

def test_agent_calculate_tool(agent):
    # This test relies on the calculate tool being available
    # and correctly configured.
    result = agent.run("What is 2 + 2?")
    assert "4" in result

def test_agent_rag_retrieval(agent):
    # This test relies on the RAG retrieval being available
    # and correctly configured.  It also requires that the agent
    # has ingested some data into the RAG system.
    agent._ingest_to_rag("test_query", "test_result", "test_tool")
    result = agent.run("What is test_query?")
    assert "test_result" in result