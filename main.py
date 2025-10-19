import os
import time
import math
import logging
from datetime import datetime, timedelta, timezone, time as dtime
from typing import List, Dict, Iterable
from zoneinfo import ZoneInfo


import numpy as np
import pandas as pd


from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, AssetClass
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest


from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# -----------------------
# Config
# -----------------------
API_KEY = os.environ.get("APCA_API_KEY_ID")
API_SECRET = os.environ.get("APCA_API_SECRET_KEY")
PAPER = os.environ.get("APCA_PAPER", "true").lower() in ("1", "true", "yes")


RUN_EVERY_SECONDS = int(os.environ.get("RUN_EVERY_SECONDS", "3600"))
BARS_NEEDED = 260


SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
ETF_ALLOWLIST = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "ARKK"]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("alpaca-hourly-bot")


ET_TZ = ZoneInfo("America/New_York")


# -----------------------
# Helpers
# -----------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
delta = series.diff()
gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
rs = gain / loss.replace(0, np.nan)
return 100 - (100 / (1 + rs))




def compute_signals(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
out: Dict[str, Dict[str, float]] = {}
for symbol, sdf in df.groupby(level=0):
closes = sdf['close']
ma60 = closes.rolling(window=60, min_periods=60).mean()
main()
