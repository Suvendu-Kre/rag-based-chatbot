# Placeholder for workflow logic.  Currently, the agent runs a simple loop.
async def run_workflow(input_message: str, agent):
    """
    Executes the default agent workflow.
    """
    return await agent.chat(input_message)