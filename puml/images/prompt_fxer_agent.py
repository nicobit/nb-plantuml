from agno.agent import Agent
from langgraph.graph import State
from openai import OpenAI
from agno.tools.base import Tool
from prompt_utils import read_prompt, save_prompt

class SaveTool(Tool):
    name = "SavePromptTool"
    description = "Saves a revised prompt with metadata."

    def __call__(self, agent_name: str, revised_prompt: str, reason: str):
        save_prompt(agent_name, revised_prompt, reason)

class PromptFixerAgent(Agent):
    def __init__(self, name="prompt_fixer_agent"):
        self.llm = OpenAI(model="gpt-4o")
        self.instructions = (
            "You are a PromptFixerAgent. Your task is to improve broken prompts. "
            "You will be given the current prompt and an error message. Suggest a better version."
        )
        self.tools = [SaveTool()]
        super().__init__(name=name, llm=self.llm, instructions=self.instructions, tools=self.tools)

    def run(self, state: State) -> State:
        agent_name = state.get("failed_agent", "")
        error_message = state.get("error_message", "")
        current_prompt = read_prompt(agent_name)

        prompt = f"""Fix this prompt for agent '{agent_name}' based on the following error:\n
---\n
PROMPT:\n{current_prompt}\n
---\n
ERROR:\n{error_message}\n
Provide the full revised prompt only."""
        revised = self.llm(prompt)
        save_prompt(agent_name, revised, reason="fix_by_agent", error=error_message)
        state["fixed_prompt"] = revised
        return state
