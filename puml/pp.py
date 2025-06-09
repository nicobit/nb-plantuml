# 📦 STEP 0 — Install Dependencies
# !pip install -U agnos-ai pandas numpy scikit-learn matplotlib seaborn lightgbm joblib

# 🧠 STEP 1 — Imports & Setup
import os, json, datetime, asyncio
from pathlib import Path
from agnos import Agent, Workspace, TaskExecutor
from agnos.client import OpenAIChatCompletionClient
from agnos.token_providers import AzureTokenProvider

# 🛡️ STEP 2 — Environment & Azure Config
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "your-azure-key")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://your-endpoint.openai.azure.com")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")

client = OpenAIChatCompletionClient(
    api_key=AZURE_API_KEY,
    endpoint=AZURE_ENDPOINT,
    deployment=AZURE_DEPLOYMENT,
    token_provider=AzureTokenProvider()
)

workspace = Workspace()

# 📘 STEP 3 — Logging Utility
def log_to_submission(agent_name, prompt, output):
    log_path = Path("submission_log.json")
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log[agent_name] = {
        "prompt": prompt,
        "output_log": output.strip(),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    log_path.write_text(json.dumps(log, indent=2))

# 🧾 STEP 4 — Prompts
prompts = {
    "EDA_Agent": """
You are EDA_Agent.
1. Load train.csv, val.csv, test.csv from /data
2. Confirm 'Close' exists
3. Compute Target_Return = log(C[t+1]) - log(C[t])
4. Save script EDA.py with summary stats, plots, correlation matrix
5. Run and debug until it works
Reply “EDA complete”
""",
    "FeatureEngineering_Agent": """
You are FeatureEngineering_Agent.
1. Load the datasets with Target_Return
2. Engineer at least 8 features (rolling avg, volume, RSI, ATR, sin/cos day, lag)
3. Save output to features.csv and script FEATURE.py
4. Run/debug until it works
Reply “Features ready”
""",
    "Modelling_Agent": """
You are Modelling_Agent.
1. Load features.csv
2. Split train/val based on data/train.csv and data/val.csv row counts
3. Train at least 3 models (RF, GBT, LightGBM)
4. Pick best using RMSE on validation
5. Save best model to model.pkl and script MODEL.py
Reply “Model trained”
""",
    "Evaluation_Agent": """
You are Evaluation_Agent.
1. Load model.pkl
2. Predict on data/test.csv, compute RMSE
3. Save script EVAL.py and result to MSFT_Score.txt: “RMSE: <float>”
4. Log result in submission_log.json
Reply “Evaluation done”
"""
}

# 🤖 STEP 5 — Define Agents
agents = {
    name: Agent(name=name, model=client, description=desc)
    for name, desc in prompts.items()
}

# 🧪 STEP 6 — Run Agents Sequentially
async def run_agent(agent, task_msg):
    print(f"▶ Running {agent.name}...")
    result = await agent.run(task=task_msg, workspace=workspace)
    log_to_submission(agent.name, agent.description, result.message.content)
    print(f"✅ {agent.name} done.\n")

# 🧵 STEP 7 — Run Pipeline
async def main():
    await run_agent(agents["EDA_Agent"], "Start EDA now")
    await run_agent(agents["FeatureEngineering_Agent"], "Generate features")
    await run_agent(agents["Modelling_Agent"], "Train models and save best one")
    await run_agent(agents["Evaluation_Agent"], "Evaluate final model")

await main()
