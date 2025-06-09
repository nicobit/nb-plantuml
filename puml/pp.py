import os
import json
from pathlib import Path
import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.local_file_system import LocalFileSystemTools
from agno.tools.python import PythonTools

# === Setup ===
os.environ["OPENAI_API_KEY"] = "sk-YOUR-KEY"

ROOT_DIR = Path.cwd()
DATA_DIR = ROOT_DIR / "data"
LOG_FILE = ROOT_DIR / "submission_log.json"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

file_tools = LocalFileSystemTools()
python_tool = PythonTools()

script_files = {
    "EDA_Agent": "EDA.py",
    "FeatureEngineering_Agent": "FEATURE.py",
    "Modelling_Agent": "MODEL.py",
    "Evaluation_Agent": "EVAL.py",
}

submission_log = {}

PROMPTS = {
    "EDA_Agent": f"""
You are EDA_Agent.

Use the following dataset paths:
- Train: {TRAIN_PATH}
- Validation: {VAL_PATH}
- Test: {TEST_PATH}

Steps:
1. Load all 3 datasets.
2. Confirm 'Close' column exists. If not, raise an error.
3. Compute Target_Return = np.log(df["Close"].shift(-1)) - np.log(df["Close"])
4. Save and run a Python script called EDA.py that:
   - prints head, tail, missing values, and basic stats
   - plots price & volume history (save as PNG)
   - prints correlation matrix with OHLCV and Target_Return

After success, reply with “EDA complete”.
""",

    "FeatureEngineering_Agent": f"""
You are FeatureEngineering_Agent.

Steps:
1. Read the datasets with Target_Return.
2. Engineer at least 8 features including:
   - log_volume = np.log1p(Volume)
   - rolling_mean_3 = df["Close"].rolling(3).mean()
   - rolling_std_5 = df["Close"].rolling(5).std()
   - RSI
   - ATR
   - day_of_week_sin/cos
   - lagged returns
3. ⚠️ Do not overwrite train.csv, val.csv, or test.csv.
4. Save features to a new file named features.csv and a script FEATURE.py.
5. Run and debug FEATURE.py until success.

Reply with “Features ready”.
""",

    "Modelling_Agent": f"""
You are Modelling_Agent.

Steps:
1. Load features.csv.
2. Split based on:
   - Train index = length of train.csv
   - Validation index = length of val.csv
3. Train 3 models:
   - RandomForestRegressor
   - GradientBoostingRegressor
   - LightGBM (if available)
4. Use RMSE on validation to select best.
5. Save best model as model.pkl in working dir.
6. Create MODEL.py and debug until complete.

Reply with “Model trained”.
""",

    "Evaluation_Agent": f"""
You are Evaluation_Agent.

Steps:
1. Load model.pkl from working dir.
2. Load test set from {TEST_PATH} and compute Target_Return.
3. Use model to predict and compute RMSE.
4. Save EVAL.py and output MSFT_Score.txt with: RMSE: <value>
5. Append RMSE to submission_log.json.

Reply with “Evaluation done”.
"""
}

def make_agent(name, instructions):
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        name=name,
        instructions=instructions,
        tools=[file_tools, python_tool],
        debug_mode=True
    )

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
        submission_log[script_key] = script_path.read_text().strip()
    else:
        submission_log[script_key] = "<script not found>"

async def main():
    await run_agent("EDA_Agent", "EDA_Script")
    await run_agent("FeatureEngineering_Agent", "FeatureEngineering_Script")
    await run_agent("Modelling_Agent", "Modeling_Script")
    await run_agent("Evaluation_Agent", "Evaluation_Script")
    LOG_FILE.write_text(json.dumps(submission_log, indent=2))
    print(f"\n✅ submission_log.json written to: {LOG_FILE}")
