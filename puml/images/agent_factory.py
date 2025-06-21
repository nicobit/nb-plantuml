from agno.agent import Agent
from prompt_utils import read_prompt, revise_prompt_on_error, save_prompt
from agno.tools.python_tool import PythonCodeTool
from tools.prompt_fixer_tool import PromptFixerTool
from openai import OpenAI
from langgraph.graph import State

class AgentWithAutoFix(Agent):
    def __init__(self, name, model="gpt-4o", placeholders: dict = None):
        self.name = name
        llm = OpenAI(model=model)
        raw_prompt = read_prompt(name)

        if placeholders:
            for key, value in placeholders.items():
                raw_prompt = raw_prompt.replace(f"{{{{{key}}}}}", value)

        self.instructions = raw_prompt
        self.tools = [PythonCodeTool(), PromptFixerTool()]
        super().__init__(name=name, llm=llm, instructions=self.instructions, tools=self.tools)

    def run(self, state: State) -> State:
        try:
            response = self.chat(state)
            with open(f"{self.name}/output.txt", "w") as f_out:
                f_out.write(response)
            state[self.name] = response
        except Exception as e:
            current_prompt = read_prompt(self.name)
            revised = revise_prompt_on_error(self.name, current_prompt, str(e))
            save_prompt(self.name, revised, reason="auto_llm_fix", error=str(e))
            state[self.name] = f"ERROR: {str(e)}\nRevised prompt saved."
        return state

def create_agent(name: str, model: str = "gpt-4o", placeholders: dict = None) -> Agent:
    return AgentWithAutoFix(name=name, model=model, placeholders=placeholders)
