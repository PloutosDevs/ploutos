import requests
from time import sleep

from source.utils import get_secrets


class CoinMarketCap:
    """
    Connector class for working with CoinMarketCap API

    https://coinmarketcap.com/api/documentation/v1
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-CMC_PRO_API_KEY": get_secrets("COIN_MARKET_CAP_PROD")
            }
        )

    def send_request(self, method, request_url, **request_params):
        """
        Send requests and processes errors

        params:
            method: GET, POST, PUT, DELETE
            request_url: Request URL
            request_params: Set of params for request: json, params etc.
        return:
            Response of request if status code == 200
        """

        request = self.session.prepare_request(requests.Request(method, request_url, **request_params))

        for _ in range(3):

            response = self.session.send(request)

            if response.status_code == 429:  # Too many requests
                sleep(60)
            else:
                break

        if response.status_code == 200:
            response = response.json()
        else:
            raise requests.ConnectionError(f"Error: {response.text}\nStatus Code: {response.status_code}")

        return response
