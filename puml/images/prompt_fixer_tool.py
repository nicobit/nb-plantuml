from agno.tools.base import Tool
from prompt_utils import revise_prompt_on_error, read_prompt

class PromptFixerTool(Tool):
    name = "PromptFixerTool"
    description = "Tool to revise the current prompt based on an error message."

    def __call__(self, agent_name: str, error_message: str) -> str:
        current_prompt = read_prompt(agent_name)
        revised = revise_prompt_on_error(agent_name, current_prompt, error_message)
        return revised
