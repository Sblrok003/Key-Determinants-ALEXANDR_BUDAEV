from pathlib import Path

import pandas as pd


START_DATE = "2015-01-01"
END_DATE = "2026-03-31"

OUTPUT_FILE = "btc_market_cap_coinmetrics_2015_2026_march.xlsx"


def load_btc_market_cap_coinmetrics() -> pd.DataFrame:
    """
    Скачивает BTC Market Cap из Coin Metrics GitHub CSV.

    Нужная метрика:
    CapMrktCurUSD = текущая рыночная капитализация BTC в USD.
    """

    url = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"

    print("Скачиваю BTC данные из Coin Metrics...")
    df = pd.read_csv(url)

    df.columns = [str(c).strip() for c in df.columns]

    print("Всего колонок:", len(df.columns))

    if "time" not in df.columns:
        raise ValueError(f"Не нашёл колонку time. Колонки: {df.columns.tolist()}")

    if "CapMrktCurUSD" not in df.columns:
        print("\nКолонки в файле:")
        print(df.columns.tolist())
        raise ValueError("Не нашёл колонку CapMrktCurUSD в Coin Metrics BTC CSV.")

    result = df[["time", "CapMrktCurUSD"]].copy()

    result = result.rename(
        columns={
            "time": "date",
            "CapMrktCurUSD": "BTC Market Cap\n(млрд $)",
        }
    )

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    result["BTC Market Cap\n(млрд $)"] = pd.to_numeric(
        result["BTC Market Cap\n(млрд $)"],
        errors="coerce",
    ) / 1_000_000_000

    result = result.dropna(subset=["date", "BTC Market Cap\n(млрд $)"])
    result = result.drop_duplicates(subset=["date"])
    result = result.sort_values("date")

    result = result[
        (result["date"] >= pd.to_datetime(START_DATE)) &
        (result["date"] <= pd.to_datetime(END_DATE))
    ].copy()

    return result


def main():
    result = load_btc_market_cap_coinmetrics()

    print("\nИТОГ:")
    print("строк:", len(result))
    print("диапазон:", result["date"].min(), "->", result["date"].max())
    print("пустых:", result["BTC Market Cap\n(млрд $)"].isna().sum())

    print("\nПервые строки:")
    print(result.head())

    print("\nПоследние строки:")
    print(result.tail())

    result.to_excel(OUTPUT_FILE, index=False)

    print("\nФайл сохранён:")
    print(Path(OUTPUT_FILE).resolve())


if __name__ == "__main__":
    main()