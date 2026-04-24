import pytest
from agents.main_agent import Agent

@pytest.fixture
def agent():
    return Agent()

def test_agent_initialization(agent):
    assert agent is not None
    assert agent.llm is not None
    assert agent.tools_map is not None

def test_agent_run_with_date_tool(agent):
    response = agent.run("What is the current date?")
    assert "current date" in response.lower()

def test_agent_run_with_unknown_query(agent):
    response = agent.run("Tell me about the capital of France.")
    assert isinstance(response, str)