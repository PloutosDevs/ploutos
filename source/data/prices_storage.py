from time import sleep
from threading import Thread
import pandas as pd

from source.data.get.binance_prices import compose_binance_candles_df

import config


class PricesStorage:
    """
    Class for storaging, gathering and updating candles data from exchanges

    params:
        start_dttm - Datetime in format "%Y-%m-%d %H:%M:%S.%f"
        end_dttm - Datetime in format "%Y-%m-%d %H:%M:%S.%f" or None
        interval - Interval of candles. All values can be seen in utils.get_time_slide_window
        tickers - Dict with keys - source (exchange) and keys - list of tickers
        auto_update - Flag for autoupdateing prices per specified period
        update_every - Period in seconds of updating candles
        time_zone - Time zone for dates
    """

    def __init__(
        self,
        start_dttm: str,
        end_dttm: str = None,
        interval: str = '5m',
        tickers: dict = {},
        auto_update: bool = True,
        update_every: int = 300,
        time_zone: str = config.DEFAULT_TZ
    ):

        self.candles = pd.DataFrame({}, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])
        self.candles.index.name = 'Time'

        self.time_zone = time_zone
        self.start_dttm = pd.Timestamp(start_dttm, tz=self.time_zone)
        self.end_dttm = pd.Timestamp(end_dttm, tz=self.time_zone) if end_dttm else None
        self.interval = interval
        self.tickers = tickers
        self.auto_update = auto_update
        self.update_every = update_every

        self._new_tickers = {}

        # Updating on background
        if self.auto_update:
            Thread(target=self.update_prices).start()

    def update_prices(self):

        while True:

            start_dttm = self.start_dttm if self.candles.empty else self.candles.index.max()

            if self.end_dttm and start_dttm > self.start_dttm:
                return

            new_candles = pd.DataFrame({}, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])
            new_candles.index.name = 'Time'

            for source, tickers in self._new_tickers.items():

                existed_tickers = self.tickers.get(source)

                if existed_tickers:

                    tickers = [ticker for ticker in tickers if ticker not in self.tickers[source]]

                    self.tickers[source] = list(set(self.tickers[source] + tickers))
                else:
                    self.tickers[source] = list(set(tickers))

            for source, tickers in self.tickers.items():

                if source == 'binance':
                    candles = compose_binance_candles_df(
                        tickers,
                        (
                            start_dttm.strftime("%Y-%m-%d %H:%M:%S.%f")
                            if tickers not in self._new_tickers.get(source, [])
                            else self.start_dttm.strftime("%Y-%m-%d %H:%M:%S.%f")
                        ),
                        end_time=self.end_dttm.strftime("%Y-%m-%d %H:%M:%S.%f") if self.end_dttm else None,
                        interval=self.interval,
                        time_zone=self.time_zone
                    )

                    new_candles = pd.concat([new_candles, candles]) if not new_candles.empty else candles

            # Drop duplicate
            self.candles = self.candles.drop(start_dttm) if not self.candles.empty else self.candles

            self.candles = pd.concat([self.candles, new_candles]) if not self.candles.empty else new_candles

            self._new_tickers = {}

            print(self.update_every)

            sleep(self.update_every)

    def add_new_tickers(self, new_tickers: dict):
        """
        Add new tickers to dict for updating

        params:
            new_tickers - Dict with keys - source (exchange) and keys - list of tickers
        """

        self._new_tickers = new_tickers

    def get_candles(self, interval: str = '1d', ticker: str = None):
        """
        Group candles by specified interval

        params:
            candles - High, Low, Close, Open, Volume, Symbol candles
            interval - Interval of candles. All values can be seen in utils.get_time_slide_window

        return:
            pd.DataFrame
        """

        resample_map = {
            '1s': 'S',  # 1 sec
            '1m': 'T',  # 1 min
            '3m': '3T',  # 3 mins
            '5m': '5T',  # 5 mins
            '15m': '15T',  # 15 mins
            '30m': '30T',  # 30 mins
            '1h': 'H',  # 1 hour
            '2h': '2H',  # 2 hours
            '4h': '4H',  # 4 hours
            '6h': '6H',  # 6 hours
            '8h': '8H',  # 8 hours
            '12h': '12H',  # 12 hours
            '1d': 'D',  # 1 day
            '3d': '3D',  # 3 days
            '1w': 'W',  # 1 week
            '1M': 'M',  # 1 month
        }

        candles = self.candles.copy()

        if ticker:
            candles = candles[candles["Symbol"] == ticker]

        if candles.empty:
            return candles

        def resample_for_symbol(group):

            # Ceil to full first interval
            first_row = group.index[0].ceil(resample_map[interval])
            group = group[group.index >= first_row]

            # Group
            return group.resample(resample_map[interval]).agg(
                {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }
            ).dropna()

        resampled_candles = candles.groupby('Symbol').apply(resample_for_symbol)

        resampled_candles.reset_index(level=0, inplace=True)

        return resampled_candles
