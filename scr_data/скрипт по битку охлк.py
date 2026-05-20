from pathlib import Path
from datetime import datetime
import time

import pandas as pd
import requests


START_DATE = "2015-01-01"
END_DATE = "2026-03-31"

OUTPUT_FILE = "btc_ohlcv_2015_2026_march.xlsx"


def to_unix(date_str: str) -> int:
    """
    Перевод даты YYYY-MM-DD в Unix timestamp.
    """
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def download_btc_yahoo() -> pd.DataFrame:
    """
    Скачивает дневные OHLCV данные BTC-USD с Yahoo Finance.
    """

    symbol = "BTC-USD"

    # Yahoo period2 не включает последнюю дату,
    # поэтому добавляем +1 день к END_DATE
    period1 = to_unix(START_DATE)
    period2 = to_unix(
        (pd.to_datetime(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print("Скачиваю BTC-USD с Yahoo Finance...")
    print("Период:", START_DATE, "->", END_DATE)

    last_error = None

    for attempt in range(1, 6):
        try:
            print(f"Попытка {attempt}/5")

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            chart = data["chart"]
            error = chart.get("error")

            if error is not None:
                raise RuntimeError(f"Yahoo вернул ошибку: {error}")

            result = chart["result"][0]

            timestamps = result["timestamp"]
            quote = result["indicators"]["quote"][0]

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        timestamps,
                        unit="s",
                        utc=True,
                    ).tz_localize(None).normalize(),

                    "BTC Open": quote["open"],
                    "BTC High": quote["high"],
                    "BTC Low": quote["low"],
                    "BTC Close": quote["close"],
                    "Объём торгов ($)": quote["volume"],
                }
            )

            df = df.dropna(subset=["date", "BTC Close"])
            df = df.drop_duplicates(subset=["date"])
            df = df.sort_values("date")

            df = df[
                (df["date"] >= pd.to_datetime(START_DATE)) &
                (df["date"] <= pd.to_datetime(END_DATE))
            ].copy()

            for col in [
                "BTC Open",
                "BTC High",
                "BTC Low",
                "BTC Close",
                "Объём торгов ($)",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            last_error = e
            print("Ошибка:", e)
            time.sleep(attempt * 3)

    raise RuntimeError(f"Не удалось скачать данные BTC: {last_error}")


def main():
    btc = download_btc_yahoo()

    print("\nГотово.")
    print("Строк:", len(btc))
    print("Диапазон:", btc["date"].min(), "->", btc["date"].max())

    print("\nПервые строки:")
    print(btc.head())

    print("\nПоследние строки:")
    print(btc.tail())

    btc.to_excel(OUTPUT_FILE, index=False)

    print("\nФайл сохранён:")
    print(Path(OUTPUT_FILE).resolve())


if __name__ == "__main__":
    main()