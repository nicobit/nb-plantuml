import os
import json
import datetime
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.local_file_system import LocalFileSystemTools
from agno.tools.python import PythonTools  # 👈 using inline Python execution

# === Setup ===
os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_FILE = ROOT_DIR / "submission_log.json"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

file_tools = LocalFileSystemTools()
python_tool = PythonTools()

# === Script file map
script_files = {
    "EDA_Agent": "EDA.py",
    "FeatureEngineering_Agent": "FEATURE.py",
    "Modelling_Agent": "MODEL.py",
    "Evaluation_Agent": "EVAL.py",
}

# === Logging structure
submission_log = {}

# === Prompts with real paths
PROMPTS = {
    "EDA_Agent": f"""
You are EDA_Agent.

Use the following dataset paths:
- Train: {TRAIN_PATH}
- Validation: {VAL_PATH}
- Test: {TEST_PATH}

Steps:
1. Load all 3 datasets and compute Target_Return = log(C[t+1]) - log(C[t]).
2. Save and run a Python script called EDA.py that:
   - prints head, tail, missing values, and basic stats
   - plots price & volume history (save as PNG)
   - prints correlation matrix with OHLCV and Target_Return
3. If execution fails, fix and retry.

After success, reply with “EDA complete”.
""",
    "FeatureEngineering_Agent": f"""
You are FeatureEngineering_Agent.

Use the same dataset paths as EDA_Agent:
- Train: {TRAIN_PATH}
- Validation: {VAL_PATH}
- Test: {TEST_PATH}

Steps:
1. Engineer at least 5 meaningful features (e.g., log-volume, rolling mean, RSI, ATR, day-of-week sine/cos).
2. Ensure no future leakage.
3. Create FEATURE.py and features.csv in working directory.
4. Run and debug FEATURE.py until success.

After success, reply with “Features ready”.
""",
    "Modelling_Agent": f"""
You are Modelling_Agent.

Use features.csv produced by FeatureEngineering_Agent.
You must:
1. Split data based on:
   - Train: {TRAIN_PATH}
   - Validation: {VAL_PATH}
2. Train at least two models (suggested: GradientBoostingRegressor and RandomForestRegressor).
3. Choose best by RMSE on validation set.
4. Save best model as model.pkl.
5. Generate MODEL.py that encapsulates full training logic.
6. Run and debug MODEL.py until complete.

After success, reply with “Model trained”.
""",
    "Evaluation_Agent": f"""
You are Evaluation_Agent.

Steps:
1. Load model.pkl and test data from {TEST_PATH}.
2. Predict next-day log return.
3. Calculate RMSE and save it to MSFT_Score.txt (format: RMSE: <float>)
4. Save EVAL.py with reproducible code, and run/debug it.
5. Append RMSE to submission_log.json.

After success, reply with “Evaluation done”.
"""
}

# === Agent factory
def make_agent(name, instructions):
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        name=name,
        instructions=instructions,
        tools=[file_tools, python_tool],
        debug_mode=True
    )

# === Agent execution helper
async def run_agent(agent_name, script_key):
    prompt = PROMPTS[agent_name]
    agent = make_agent(agent_name, prompt)
    print(f"\n=== Running {agent_name} ===")
    result = await agent.run("Start now.")
   
    submission_log[agent_name] = {
        "prompt": prompt.strip(),
        "output_log": result.strip()
    }

    script_path = ROOT_DIR / script_files[agent_name]
    if script_path.exists():
        script_content = script_path.read_text()
        submission_log[script_key] = script_content.strip()
    else:
        submission_log[script_key] = "<script not found>"

# === Main orchestrator
async def main():
    await run_agent("EDA_Agent", "EDA_Script")
    await run_agent("FeatureEngineering_Agent", "FeatureEngineering_Script")
    await run_agent("Modelling_Agent", "Modeling_Script")
    await run_agent("Evaluation_Agent", "Evaluation_Script")

    LOG_FILE.write_text(json.dumps(submission_log, indent=2))
    print(f"\n✅ submission_log.json written to: {LOG_FILE}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
