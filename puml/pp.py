from agnos import Agent, Workspace, TaskExecutor
from agnos.client import OpenAIChatCompletionClient
from agnos.token_providers import AzureTokenProvider
import os

# 🌐 Azure OpenAI Setup
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT")  # e.g., "gpt-4"

# Token provider with default Azure credentials (interactive, environment or managed identity)
token_provider = AzureTokenProvider()

client = OpenAIChatCompletionClient(
    api_key=AZURE_API_KEY,
    endpoint=AZURE_ENDPOINT,
    deployment=AZURE_DEPLOYMENT,
    token_provider=token_provider
)

workspace = Workspace()

# Define the four agents using prompts only
eda_agent = Agent(
    name="EDA_Agent",
    model=client,
    description="""You are an expert data scientist. Your job is to perform Exploratory Data Analysis on MSFT stock data 
    (train.csv, val.csv, test.csv in /data). Create a script named EDA.py. It must load train.csv, describe statistics, plot trends,
    check missing values, and save EDA.py in the working directory. Execute and debug the script if needed."""
)

feature_agent = Agent(
    name="FeatureEngineering_Agent",
    model=client,
    description="""You are a feature engineering expert. Your task is to create FEATURE.py that loads EDA-transformed data,
    computes next-day log returns as Target_Return, adds technical indicators like moving averages or volume changes,
    and saves the script as FEATURE.py. Execute and debug it."""
)

model_agent = Agent(
    name="Modelling_Agent",
    model=client,
    description="""You are an expert ML engineer. Your job is to write MODEL.py that trains a regression model on the training set
    (with features from FEATURE.py), validates on val.csv, and saves the model. Use sklearn or XGBoost. Execute and debug it."""
)

eval_agent = Agent(
    name="Evaluation_Agent",
    model=client,
    description="""You are an evaluator. Create EVAL.py which loads the model from MODEL.py, predicts log returns on test.csv,
    computes RMSE, and writes it as: 'RMSE: <float>' to MSFT_Score.txt. Save a JSON log as submission_log.json with step details."""
)
