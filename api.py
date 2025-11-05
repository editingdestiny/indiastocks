from fastapi import FastAPI
import pandas as pd
from pathlib import Path

app = FastAPI()

@app.get("/api/")
async def read_root():
    return {"message": "Welcome to India Stock API"}

@app.get("/api/data")
async def get_stock_data():
    try:
        # Adjust the path based on where your data is stored
        df = pd.read_csv("nse_all_10y.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}