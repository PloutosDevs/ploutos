from source.data.get.binance_prices import get_binance_symbols, compose_binance_candles_df
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
import config 
import os

# Подключение к базе данных (или создание файла, если его нет)
# Создайте подключение к базе данных SQLite
engine = create_engine(f"sqlite:///{os.path.join(config.DATA_PATH, 'ploutos_data.db')}")

valuation_date = pd.Timestamp.today(tz=config.DEFAULT_TZ).normalize()
candles_period = 90
candles_start_date = (valuation_date - pd.Timedelta(days=candles_period)).strftime("%Y-%m-%d")
candles_end_date = valuation_date.strftime("%Y-%m-%d") + " 23:59:59.999999"

# BINANCE_SYMBOLS = get_binance_symbols(only_usdt=True)
BINANCE_SYMBOLS = ['BTCUSDT', 'LOOMUSDT', 'HIFIUSDT', 'VICUSDT', 'RPLUSDT', 'ATAUSDT']

eval_candles_df = compose_binance_candles_df(BINANCE_SYMBOLS, 
                                             start_time=candles_start_date,
                                             end_time=candles_end_date,
                                             interval='1d')
eval_candles_df['Source'] = 'BINANCE'

eval_candles_df.to_sql(
    name="market_data",         # Имя таблицы
    con=engine,                 # Подключение к базе данных
    if_exists="append",         # Поведение при существовании таблицы: "fail", "replace", "append"
    index=True,                 # Сохранение индекса как столбца
    index_label="Time"          # Имя для индекса, если оно нужно
)

with engine.connect() as connection:
    rows = connection.execute(text("SELECT * FROM market_data limit 10"))

    for row in rows:
        print(row)