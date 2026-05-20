"""
Делает непрерывный ежедневный ряд и заполняет пропуски ffill
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(SCRIPT_DIR, "macro_sp500_dxy.xlsx")
OUTPUT = os.path.join(SCRIPT_DIR, "macro_sp500_dxy_daily.xlsx")
SHEET  = "Macro"

df = pd.read_excel(INPUT, sheet_name=SHEET, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.index.name = "Дата"

# Полный непрерывный диапазон по дням
full_range = pd.date_range(start="2015-01-01", end="2026-04-30", freq="D")

df = df.reindex(full_range)
df.index.name = "Дата"

# Заполняем пропуски предыдущим значением
df = df.ffill()

print(f"Строк: {len(df)} | {df.index.min().date()} → {df.index.max().date()}")
print(f"Пропусков осталось: {df.isna().sum().sum()}")

df.to_excel(OUTPUT, sheet_name="Macro")
print(f"✅ Готово: {OUTPUT}")