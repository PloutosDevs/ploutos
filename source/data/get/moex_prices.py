import pandas as pd
import requests

# https://fs.moex.com/files/6523 moex doc


def get_candles_moex(ticker: str, interval: int, start_time: str, end_time=None,
                     time_zone="Europe/Moscow", engine="stock", market="shares", board=None):
    """
    Return candles from MOEX exchange according to params. Can return any candles: stocks, currencies, indexes
    All engine, market and board can be found there "https://iss.moex.com/iss/index.json"

    params:
        ticker: Ticker of instrument
            Notes:
            USDRUB - USD000UTSTOM
            Index MOEX - IMOEX
        interval: Interval of candles
            Can be:
            1 - 1 min,
            10 - 10 min,
            60 - 60 min,
            24 - 1 day,
            7 - 7 days,
            31 - 31 days,
            4 - 4 months
        start_time: Start datetime of candles in format "%Y-%m-%d %H:%M:%S.%f"
        end_time: End datetime of candles in format "%Y-%m-%d %H:%M:%S.%f". Default is now
        time_zone: Your timezone pytz.all_timezones
        engine: Common used examples: "stock" for indexes and shares, "currency" for currencies
        market: Common used examples: "shares", "index" and "selt" (currencies)
        board: Not required param, MOEX use default primary board if it's None.
               Common used examples: "TQBR" for indexes and shares, "CETS" for currencies
    return:
        Candles with OHLCV cols and pd.Timestamp index
    """

    start_time = pd.Timestamp(start_time, tz=time_zone).tz_convert(tz="Europe/Moscow")

    if not end_time or pd.Timestamp(end_time, tz=time_zone) > pd.Timestamp.now(tz=time_zone):
        end_time = pd.Timestamp.now(tz=time_zone).tz_convert(tz="Europe/Moscow")
    else:
        end_time = pd.Timestamp(end_time, tz=time_zone).tz_convert(tz="Europe/Moscow")

    # Base DF
    base_candles = pd.DataFrame({}, columns=["Time", "Open", "High", "Low", "Close", "Volume"]).astype(float)

    # Base offset
    start_row = 0
    len_rows = 10e9

    # Shift the offset and get data while len of returned data is not 0
    while len_rows != 0:

        params = {
            "from": str(start_time),
            "till": str(end_time),
            "start": start_row,
            "interval": interval
        }

        # Try requests 3 times before status code == 200
        for _ in range(3):

            # If boards was entered, add param in URL
            try:
                if board:
                    response = requests.get(
                        f"http://iss.moex.com/iss/engines/{engine}/markets/{market}/boards/{board}"
                        f"/securities/{ticker}/candles.json",
                        params=params, timeout=10)
                else:
                    response = requests.get(
                        f"http://iss.moex.com/iss/engines/{engine}/markets/{market}/securities/{ticker}/candles.json",
                        params=params, timeout=10)
            except requests.ReadTimeout:
                raise requests.exceptions.ConnectionError("Timeout")

            if response.status_code == 200:
                response = response.json()
                break

        if isinstance(response, requests.Response):
            raise requests.exceptions.ConnectionError(response.text)

        if response["candles"]["data"]:
            candles = pd.DataFrame(response["candles"]["data"], columns=response["candles"]["columns"])

            candles = candles[["open", "close", "high", "low", "value", "begin"]]
            candles.columns = ["Open", "Close", "High", "Low", "Volume", "Time"]

            candles["Time"] = pd.DatetimeIndex(candles["Time"], tz="Europe/Moscow").tz_convert(tz=time_zone)

            base_candles = pd.concat([base_candles, candles])

            len_rows = len(candles)
        else:
            len_rows = 0

        start_row += len_rows

    base_candles = base_candles.drop_duplicates("Time").set_index("Time").astype(float)

    return base_candles


def get_moex_boards(engine="stock", market="shares"):
    """
    Return DataFrame with moex boards according to params

    params:
        engine: Common used examples: "stock" for indexes and shares, "currency" for currencies
        market: Common used examples: "shares", "index" and "selt" (currencies)
    return:
        MOEX boards
    """

    # Try requests 3 times before status code == 200
    for _ in range(3):
        response = requests.get(f"https://iss.moex.com/iss/engines/{engine}/markets/{market}/boards.json")

        if response.status_code == 200:
            response = response.json()
            break

    if isinstance(response, requests.Response):
        raise requests.exceptions.ConnectionError(response.text)

    boards = pd.DataFrame(response["boards"]["data"], columns=response["boards"]["columns"])

    return boards


def get_moex_securities(engine="stock", market="shares", boards=["TQBR", "TQPI"]):
    """
    Return DataFrame with instruments according to params

    params:
        engine: Common used examples: "stock" for indexes and shares, "currency" for currencies
        market: Common used examples: "shares", "index" and "selt" (currencies)
        board:  Exchange trading mode. Common used examples: "TQBR" for indexes and shares, "CETS" for currencies
    return:
        MOEX instruments
    """

    securities_info = []

    for board in boards:

        # Try requests 3 times before status code == 200
        for _ in range(3):
            response = requests.get(
                f"https://iss.moex.com/iss/engines/{engine}/markets/{market}/boards/{board}/securities.json"
            )

            if response.status_code == 200:
                response = response.json()
                break

        if isinstance(response, requests.Response):
            raise requests.exceptions.ConnectionError(response.text)

        moex_securities = pd.DataFrame(response["securities"]["data"], columns=response["securities"]["columns"])
        securities_info.append(moex_securities.copy())

    securities_info = pd.concat(securities_info)

    return securities_info
