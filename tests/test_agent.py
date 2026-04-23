import pytest
from agents.main_agent import Agent

def test_agent_initialization():
    agent = Agent()
    assert agent is not None
    assert agent.system_prompt is not None

def test_agent_run_basic_query():
    agent = Agent()
    response = agent.run("What is a RAG pipeline?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_agent_run_with_tool_call():
    agent = Agent()
    response = agent.run("What is 2 + 2?")
    assert isinstance(response, str)
    assert "4" in response # Expect the agent to use the calculator tool