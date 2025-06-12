
## 2 project tree

```
fast_agno_msft/
├── run_pipeline.py          # orchestrator (50 lines)
├── agent_prompts/
│   ├── EDA_prompt.txt
│   ├── FEATURE_prompt.txt
│   ├── MODEL_prompt.txt
│   └── EVAL_prompt.txt
└── data/                    # train / val / test already provided
```

You only edit **run\_pipeline.py** and the **four prompt files**.

---

## 3 Prompt snippets (copy/paste)

### `agent_prompts/EDA_prompt.txt`

```
You are **EDA_Agent**, an expert time-series analyst.

Goal ▸ write EDA.py, run it, debug until “EDA completed”.

Must:
1. Load train/val/test from /data, parse Date, sort.
2. Print shape, date span, df.describe(), NaN counts.
3. Quick checks only (no plots): print max/min Close + Volume rows.
4. Show correlation matrix (Open, High, Low, Close, Volume, Target_Return).
5. Print one insight per bullet, then “EDA completed”.
```

*(No plots keeps runtime <1 s.)*

---

### `agent_prompts/FEATURE_prompt.txt`

```
You are **FeatureEngineering_Agent**.

Goal ▸ create FEATURE.py, run it, debug until “Feature engineering completed”.

Mandatory indicators (no look-ahead):
• MA_5, MA_10
• Return_1d, 3d, 7d
• Volatility_5d, 10d
• RSI_14
• Volume_AVG_5, Volume_Change_1d, OBV
• High_Low_Range, Close_Open_Diff, Pct_Close_Open

Add boosters:
• PrevRet_1  (one-day lag of Target_Return; name MUST NOT contain the word “target”)
• MACD (EMA12−EMA26) and MACD_signal (EMA9 of MACD)
• DayOfWeek_sin, DayOfWeek_cos  (encode calendar effect)

Processing rules:
• Use pandas rolling/ewm; NaNs from initial windows — keep and let model ignore.
• **Do not scale** (tree model later).
• After computing new columns, set `Date` as the index (`df.set_index('Date', inplace=True)`)
  **and leave it OUT of the saved *_features.csv** (include only numeric predictors + Target_Return).

Save *_features.csv (train/val/test) incl. Target_Return.  
Print list(df.columns) and finish with “Feature engineering completed”.
```

---

### `agent_prompts/MODEL_prompt.txt`

```
You are **Modelling_Agent**.

Goal ▸ write MODEL.py, run it, debug until “Model training completed”.

Workflow:
1. Read train_features.csv & validation_features.csv.
2. Ensure no non-numeric columns remain:  
3. Drop rows with NaNs in X or y.
4. TimeSeriesSplit(n_splits=3, gap=0).  
   Evaluate RMSE via cross_val_score (neg_root_mean_squared_error).
5. Models:
   • LinearRegression (baseline)
   • RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
   • LightGBMRegressor(objective='regression', learning_rate=0.05,
       n_estimators=1_000, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
       random_state=42, early_stopping_rounds=50)
6. Pick lowest mean-CV RMSE.  If >0.010, build weight-0.8 LightGBM + 0.2 LinearRegression ensemble.
7. Retrain winner on train+val combined; save best_model.pkl with joblib.
8. Print “Best model: …, CV-RMSE = …” then “Model training completed”.
```

*(LightGBM fallback: if `import lightgbm` fails, switch to `HistGradientBoostingRegressor`.)*

---

### `agent_prompts/EVAL_prompt.txt`

```
You are **Evaluation_Agent**.

Goal ▸ write EVAL.py, run it, debug until “Evaluation completed”.

1. Load test_features.csv; X_test, y_test.
2. Load best_model.pkl.  If dict with 'weights', blend predictions.
3. y_pred = model.predict(X_test)
4. rmse = sqrt(mean_squared_error(y_test, y_pred))
5. Write MSFT_Score.txt  with exactly:  RMSE: {rmse:.8f}
6. Append to submission_log.json:
   {"EDA_Agent": ts, "FeatureEngineering_Agent": ts, "Modelling_Agent": ts,
    "Evaluation_Agent": ts, "RMSE": rmse}
7. Print “Final Test RMSE: … | Evaluation completed”.
```

---

## 4 Orchestrator in Agno (`run_pipeline.py`)

```python
from agno import Agent, Sequential
from pathlib import Path
import subprocess, textwrap, json, time, uuid, os, tempfile

ROOT = Path(__file__).parent
P = lambda name: (ROOT / "agent_prompts" / f"{name}_prompt.txt").read_text()

def write_and_run(code_str, filename):
    Path(filename).write_text(code_str)
    subprocess.run(["python", filename], check=True)

def make_agent(name):
    prompt = P(name)
    return Agent(
        name=name,
        system_prompt=prompt,
        on_message=lambda msg: write_and_run(msg.content, f"{name.split('_')[0]}.py"),
    )

# build flow
flow = Sequential(
    make_agent("EDA"),
    make_agent("FEATURE"),
    make_agent("MODEL"),
    make_agent("EVAL"),
)

if __name__ == "__main__":
    tic = time.time()
    flow.run()                     # 4 LLM calls + 4 script execs
    print(f"⏱  total wall-clock {time.time()-tic:.1f}s")
    print((ROOT / "MSFT_Score.txt").read_text())
```

**That’s < 50 LoC**; no helper functions sneak into agents—the code block lives only in the orchestrator.

---

## 5 One-command execution

```bash
pip install agno lightgbm joblib pandas numpy scikit-learn
python run_pipeline.py
```

