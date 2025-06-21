import os
import json
from datetime import datetime
from openai import OpenAI

def read_prompt(agent_name):
    path = f"{agent_name}/prompt.txt"
    with open(path, "r") as f:
        return f.read()

def save_prompt(agent_name, new_prompt, reason="manual", score=None, error=None):
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    prompt_path = f"{agent_name}/prompt_{version}.txt"
    with open(prompt_path, "w") as f:
        f.write(new_prompt)

    # Overwrite main prompt
    with open(f"{agent_name}/prompt.txt", "w") as f:
        f.write(new_prompt)

    log_entry = {
        "agent": agent_name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "origin": reason,
        "score": score,
        "error": error,
    }

    registry_file = "prompt_registry.json"
    if not os.path.exists(registry_file):
        with open(registry_file, "w") as f:
            json.dump([], f)

    with open(registry_file, "r") as f:
        registry = json.load(f)

    registry.append(log_entry)

    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)

def revise_prompt_on_error(agent_name, current_prompt, error_message, model="gpt-4o"):
    llm = OpenAI(model=model)

    system_message = (
        "You are a helpful prompt engineer. Improve the prompt below so the agent does not fail. "
        "You will be given the previous prompt and an error message. Provide only the revised prompt."
    )

    user_message = f"""PREVIOUS PROMPT:
{current_prompt}

ERROR:
{error_message}

Revised Prompt:
"""

    revised = llm(system_message=system_message, prompt=user_message)
    save_prompt(agent_name, revised, reason="auto_llm_fix", error=error_message)
    return revised
