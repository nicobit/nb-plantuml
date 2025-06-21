from langgraph.graph import State, Graph
from agent_factory import create_agent
from prompt_fixer_agent.agent import PromptFixerAgent
from prompt_utils import revise_prompt_on_error, read_prompt

# LLM model used for fixing prompts
FIX_MODEL = "gpt-4o"

# Create agents
eda = create_agent("eda_agent")
fe = create_agent("feature_engineering_agent")
model = create_agent("modeling_agent")
eval = create_agent("evaluation_agent")
fixer = PromptFixerAgent()

def handle_with_retries(agent_fn, agent_name):
    def wrapper(state: State) -> State:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = agent_fn(state)
                val = result.get(agent_name, "")
                if isinstance(val, str) and val.startswith("ERROR:"):
                    raise RuntimeError(val.split("ERROR:")[1].strip())
                return result
            except Exception as e:
                print(f"[{agent_name.upper()}] Attempt {attempt+1} failed: {e}")
                current_prompt = read_prompt(agent_name)
                revised = revise_prompt_on_error(agent_name, current_prompt, str(e), model=FIX_MODEL)
                state["failed_agent"] = agent_name
                state["error_message"] = str(e)
                state[agent_name] = f"ERROR: {str(e)}\nPrompt revised with {FIX_MODEL}"
        return state
    return wrapper

graph = Graph()
graph.add_node("EDA", handle_with_retries(eda.run, "eda_agent"))
graph.add_node("FeatureEngineering", handle_with_retries(fe.run, "feature_engineering_agent"))
graph.add_node("Modeling", handle_with_retries(model.run, "modeling_agent"))
graph.add_node("Evaluation", handle_with_retries(eval.run, "evaluation_agent"))

graph.set_entry_point("EDA")
graph.add_edge("EDA", "FeatureEngineering")
graph.add_edge("FeatureEngineering", "Modeling")
graph.add_edge("Modeling", "Evaluation")

initial_state = State({})
final_state = graph.run(initial_state)
print("\n🎯 FINAL STATE")
print(final_state)
