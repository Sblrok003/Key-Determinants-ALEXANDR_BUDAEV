import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. НАСТРОЙКИ LINK
# ============================================================

ASSET_NAME = "LINK"

FILE_PATH = r"C:\Users\mrsas\PycharmProjects\BTC\BTC_pr\БОЛЬШАЯ БАЗА ДАННЫХ ПО ВСЕМ.xlsx"

# Проверь, что лист реально так называется.
# Если у тебя лист называется иначе, например "ML_CHAINLINK",
# поменяй только эту строку.
SHEET_NAME = "ML_LINK"

OUTPUT_DIR = Path(FILE_PATH).parent / "link_1_1_baseline_feature_test_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
FEATURE_PLOTS_DIR = PLOTS_DIR / "features"

# LINK-данные начинаются примерно с 2017-09-20.
# Поэтому train начинаем не с 2015, а с фактического старта LINK.
TRAIN_START = "2017-09-20"
TRAIN_END = "2021-12-31"

VAL_START = "2022-01-01"
VAL_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = "2026-03-31"

HORIZONS = [1, 3, 7, 14, 30]

TARGET_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05]

MIN_TRAIN_ROWS = 100
MIN_VAL_ROWS = 50
MIN_TEST_ROWS = 50

SELECTION_MODELS = [
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
]

USE_XGBOOST = True
USE_CATBOOST = True


# ============================================================
# 2. НАЗВАНИЯ КОЛОНОК LINK
# ============================================================

RENAME_MAP = {
    "Дата": "date",
    "date": "date",

    "Лог-доходность": "log_return",
    "log_return": "log_return",

    "log_return_lag_1": "log_return_lag_1",
    "log_return_lag_2": "log_return_lag_2",
    "log_return_lag_3": "log_return_lag_3",

    "atr_14": "atr_14",
    "ATR_14": "atr_14",

    "rolling_vol_7": "rolling_vol_7",
    "rolling_vol_14": "rolling_vol_14",
    "rolling_vol_30": "rolling_vol_30",

    # Для LINK правильно использовать LINK_mcap_log_return.
    # Но оставил запасной вариант ETH_mcap_log_return,
    # если в Excel колонка случайно осталась со старым названием,
    # а внутри уже лежат LINK-значения.
    "LINK_mcap_log_return": "link_mcap_log_return",
    "link_mcap_log_return": "link_mcap_log_return",
    "ETH_mcap_log_return": "link_mcap_log_return",
    "eth_mcap_log_return": "link_mcap_log_return",

    "total_mcap_log_return": "total_mcap_log_return",
    "btc_dominance_change": "btc_dominance_change",

    "RSI (14)": "rsi_14",
    "RSI_14": "rsi_14",
    "rsi_14": "rsi_14",

    "rsi_oversold_dummy": "rsi_oversold_dummy",
    "rsi_overbought_dummy": "rsi_overbought_dummy",
    "sma_ratio": "sma_ratio",

    "Индекс страха и жадности": "fear_greed_index",
    "fear_greed_index": "fear_greed_index",
    "fear_greed_change": "fear_greed_change",
    "fear_dummy": "fear_dummy",
    "greed_dummy": "greed_dummy",

    "Поисковая активность (Google Trends)": "google_trends",
    "Поисковая активность Chainlink (Google Trends)": "google_trends",
    "google_trends": "google_trends",

    "candle_body": "candle_body",
    "ohlc_range": "ohlc_range",
    "volume_growth": "volume_growth",
    "Давление покупателей/продавцов (buy/sell pressure)": "buy_sell_pressure",
    "buy_sell_pressure": "buy_sell_pressure",

    "transactions_log_return": "transactions_log_return",
    "fees_log_return": "fees_log_return",
    "active_addresses_log_return": "active_addresses_log_return",
    "avg_fee_log_return": "avg_fee_log_return",

    "sp500_return": "sp500_return",
    "dxy_return": "dxy_return",
    "Процентные ставки (ФРС)": "fed_rate",
    "fed_rate": "fed_rate",
    "Инфляция (CPI) США": "us_cpi_inflation",
    "us_cpi_inflation": "us_cpi_inflation",
    "gold_log_return": "gold_log_return",

    "target_return_1d": "target_return_1d",
    "target_direction_1d": "target_direction_1d",

    "target_return_3d": "target_return_3d",
    "target_direction_3d": "target_direction_3d",

    "target_return_7d": "target_return_7d",
    "target_direction_7d": "target_direction_7d",

    "target_return_14d": "target_return_14d",
    "target_direction_14d": "target_direction_14d",

    "target_return_30d": "target_return_30d",
    "target_direction_30d": "target_direction_30d",
}


# ============================================================
# 3. ГРУППЫ ПРИЗНАКОВ LINK
# ============================================================

SET0_BASE = [
    "log_return_lag_1",
    "log_return_lag_2",
    "log_return_lag_3",
    "rolling_vol_7",
    "rolling_vol_14",
    "rolling_vol_30",
]

MARKET_FEATURES = [
    "rsi_14",
    "rsi_oversold_dummy",
    "rsi_overbought_dummy",
    "atr_14",
    "sma_ratio",
    "candle_body",
    "ohlc_range",
    "volume_growth",
    "buy_sell_pressure",
    "link_mcap_log_return",
    "total_mcap_log_return",
    "btc_dominance_change",
]

BEHAVIOR_FEATURES = [
    "fear_greed_index",
    "fear_greed_change",
    "fear_dummy",
    "greed_dummy",
    "google_trends",
]

ONCHAIN_FEATURES = [
    "transactions_log_return",
    "fees_log_return",
    "active_addresses_log_return",
    "avg_fee_log_return",
]

MACRO_FEATURES = [
    "sp500_return",
    "dxy_return",
    "gold_log_return",
    "fed_rate",
    "us_cpi_inflation",
]

FEATURE_GROUPS = {
    "market": MARKET_FEATURES,
    "behavior": BEHAVIOR_FEATURES,
    "onchain": ONCHAIN_FEATURES,
    "macro": MACRO_FEATURES,
}

ALL_DETERMINANTS = (
    MARKET_FEATURES
    + BEHAVIOR_FEATURES
    + ONCHAIN_FEATURES
    + MACRO_FEATURES
)


# ============================================================
# 4. СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_col_name(name):
    name = str(name)
    name = name.replace("\n", " ").replace("\r", " ")
    name = name.replace('"', "").replace("'", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def fix_duplicate_target_columns(cols):
    fixed = []
    target_seen = {}

    for col in cols:
        c = clean_col_name(col)
        base = re.sub(r"\.\d+$", "", c)

        m = re.match(r"^target_return_(\d+)d$", base)

        if m:
            h = m.group(1)

            if base not in target_seen:
                target_seen[base] = 0
                fixed.append(f"target_return_{h}d")
            else:
                target_seen[base] += 1
                fixed.append(f"target_direction_{h}d")
        else:
            fixed.append(c)

    return fixed


def collapse_duplicate_columns(df):
    out = {}

    for i, col in enumerate(df.columns):
        s = df.iloc[:, i]
        if col in out:
            out[col] = out[col].combine_first(s)
        else:
            out[col] = s

    return pd.DataFrame(out)


def to_numeric_safe(s):
    if pd.api.types.is_numeric_dtype(s):
        return s

    s = s.astype(str)
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("−", "-", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def safe_auc(y_true, proba):
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return roc_auc_score(y_true, proba)
    except Exception:
        return np.nan


def calc_metrics(y_true, pred, proba=None):
    result = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }

    if proba is not None:
        result["roc_auc"] = safe_auc(y_true, proba)
    else:
        result["roc_auc"] = np.nan

    return result


def get_positive_proba(model, X):
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X)

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)

    if classes is None:
        if proba.shape[1] == 2:
            return proba[:, 1]
        return np.full(len(X), np.nan)

    classes = list(classes)

    if 1 in classes:
        return proba[:, classes.index(1)]

    return np.zeros(len(X))


def safe_filename(name):
    name = str(name)
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я_\-]+", "_", name)
    name = name.strip("_")
    return name[:120]


def feature_group_name(feature):
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "unknown"


# ============================================================
# 5. ЗАГРУЗКА ДАННЫХ
# ============================================================

def read_excel_with_fallback():
    """
    В таблице первая строка может быть группировочной:
    Поведенческие / Целевые переменные и т.д.
    Поэтому сначала пробуем header=1.
    Если не нашли колонку Дата, пробуем header=0.
    """

    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=1)
    cols_clean = [clean_col_name(c) for c in df.columns]

    if "Дата" in cols_clean or "date" in cols_clean:
        return df

    print("WARNING: header=1 не дал колонку Дата. Пробую header=0.")
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=0)
    return df


def load_data():
    print("Loading LINK data...")

    df = read_excel_with_fallback()

    df.columns = fix_duplicate_target_columns(df.columns)

    keep_cols = []
    for c in df.columns:
        if str(c).lower().startswith("unnamed") and df[c].isna().all():
            continue
        keep_cols.append(c)

    df = df[keep_cols]

    with open(OUTPUT_DIR / "columns_before_rename.txt", "w", encoding="utf-8") as f:
        for c in df.columns:
            f.write(str(c) + "\n")

    rename_actual = {}
    for c in df.columns:
        base = clean_col_name(c)
        if base in RENAME_MAP:
            rename_actual[c] = RENAME_MAP[base]

    df = df.rename(columns=rename_actual)
    df = collapse_duplicate_columns(df)

    with open(OUTPUT_DIR / "columns_after_rename.txt", "w", encoding="utf-8") as f:
        for c in df.columns:
            f.write(str(c) + "\n")

    if "date" not in df.columns:
        raise ValueError("Не найдена колонка date. Проверь колонку 'Дата'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Отсекаем всё до фактического старта LINK.
    df = df[df["date"] >= pd.to_datetime(TRAIN_START)].copy()

    for c in df.columns:
        if c != "date":
            df[c] = to_numeric_safe(df[c])

    df = create_targets(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    df.head(300).to_csv(
        OUTPUT_DIR / "prepared_preview.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min()} — {df['date'].max()}")

    return df


def create_targets(df):
    """
    target_return_h должен быть лог-доходностью LINK за h дней вперёд.

    Для каждого горизонта создаём несколько бинарных целей:
    threshold = 0%  -> обычный рост/не рост
    threshold = 1%  -> рост больше +1% или падение ниже -1%
    threshold = 3%  -> рост больше +3% или падение ниже -3%
    """

    for h in HORIZONS:
        ret_col = f"target_return_{h}d"

        if ret_col not in df.columns:
            print(f"WARNING: no {ret_col}")
            continue

        for thr in TARGET_THRESHOLDS:
            pct = int(round(thr * 100))
            target_col = f"target_direction_{h}d_thr_{pct}pct"

            if thr == 0:
                df[target_col] = np.where(
                    df[ret_col] > 0,
                    1,
                    np.where(df[ret_col] <= 0, 0, np.nan)
                )
            else:
                df[target_col] = np.where(
                    df[ret_col] > thr,
                    1,
                    np.where(df[ret_col] < -thr, 0, np.nan)
                )

    return df


# ============================================================
# 6. МОДЕЛИ
# ============================================================

def get_models(y_train):
    models = {}

    models["Baseline_MajorityClass"] = DummyClassifier(strategy="most_frequent")

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if USE_XGBOOST:
        try:
            from xgboost import XGBClassifier

            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            models["XGBoost"] = XGBClassifier(
                n_estimators=500,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=5.0,
                reg_alpha=1.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )
        except Exception:
            pass

    if USE_CATBOOST:
        try:
            from catboost import CatBoostClassifier

            models["CatBoost"] = CatBoostClassifier(
                iterations=500,
                depth=4,
                learning_rate=0.03,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                auto_class_weights="Balanced",
                l2_leaf_reg=8.0,
            )
        except Exception:
            pass

    return models


# ============================================================
# 7. ПОДГОТОВКА ДАННЫХ
# ============================================================

def existing_features(df, features):
    return [f for f in features if f in df.columns]


def make_model_data(df, features, target):
    needed = ["date"] + features + [target]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    data = df[needed].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=features + [target])

    data = data[
        (data["date"] >= TRAIN_START) &
        (data["date"] <= TEST_END)
    ].copy()

    return data


def split_data(data):
    train = data[
        (data["date"] >= TRAIN_START) &
        (data["date"] <= TRAIN_END)
    ].copy()

    val = data[
        (data["date"] >= VAL_START) &
        (data["date"] <= VAL_END)
    ].copy()

    test = data[
        (data["date"] >= TEST_START) &
        (data["date"] <= TEST_END)
    ].copy()

    return train, val, test


def split_info(train, val, test, target):
    return {
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),

        "train_start": train["date"].min() if len(train) else pd.NaT,
        "train_end": train["date"].max() if len(train) else pd.NaT,

        "validation_start": val["date"].min() if len(val) else pd.NaT,
        "validation_end": val["date"].max() if len(val) else pd.NaT,

        "test_start": test["date"].min() if len(test) else pd.NaT,
        "test_end": test["date"].max() if len(test) else pd.NaT,

        "train_class_1_share": train[target].mean() if len(train) else np.nan,
        "validation_class_1_share": val[target].mean() if len(val) else np.nan,
        "test_class_1_share": test[target].mean() if len(test) else np.nan,
    }


def data_ok(train, val, test, target):
    if len(train) < MIN_TRAIN_ROWS:
        return False

    if len(val) < MIN_VAL_ROWS:
        return False

    if len(test) < MIN_TEST_ROWS:
        return False

    if train[target].nunique() < 2:
        return False

    if val[target].nunique() < 2:
        return False

    if test[target].nunique() < 2:
        return False

    return True


# ============================================================
# 8. ОБУЧЕНИЕ И ОЦЕНКА
# ============================================================

def fit_eval_models(data, features, target, experiment_type, feature_set_name, extra=None):
    if extra is None:
        extra = {}

    train, val, test = split_data(data)

    if not data_ok(train, val, test, target):
        return []

    rows = []

    X_train = train[features] if len(features) > 0 else np.zeros((len(train), 1))
    y_train = train[target].astype(int).values

    X_val = val[features] if len(features) > 0 else np.zeros((len(val), 1))
    y_val = val[target].astype(int).values

    X_test = test[features] if len(features) > 0 else np.zeros((len(test), 1))
    y_test = test[target].astype(int).values

    info = split_info(train, val, test, target)

    models = get_models(y_train)

    for model_name, model in models.items():
        if model_name != "Baseline_MajorityClass" and len(features) == 0:
            continue

        try:
            model.fit(X_train, y_train)
        except Exception:
            continue

        for split_name, X_eval, y_eval in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            try:
                pred = model.predict(X_eval)
                proba = get_positive_proba(model, X_eval)

                metrics = calc_metrics(y_eval, pred, proba)

                row = {
                    "asset": ASSET_NAME,
                    "experiment_type": experiment_type,
                    "feature_set_name": feature_set_name,
                    "model": model_name,
                    "eval_split": split_name,
                    "target": target,
                    "n_features": len(features),
                    "features": ",".join(features),
                    **info,
                    **metrics,
                    **extra,
                }

                rows.append(row)

            except Exception:
                continue

    return rows


# ============================================================
# 9. ЭТАП A: BASELINE И SET0
# ============================================================

def run_baseline_and_base(df):
    print("\n=== LINK 1.1A Baseline and SET0 ===")

    base_features = existing_features(df, SET0_BASE)

    all_rows = []

    for h in HORIZONS:
        for thr in TARGET_THRESHOLDS:
            pct = int(round(thr * 100))
            target = f"target_direction_{h}d_thr_{pct}pct"

            data = make_model_data(df, base_features, target)

            if data.empty:
                continue

            rows_baseline = fit_eval_models(
                data=data,
                features=[],
                target=target,
                experiment_type="baseline",
                feature_set_name="Baseline_MajorityClass",
                extra={
                    "horizon": h,
                    "target_threshold_pct": pct,
                    "target_threshold": thr,
                }
            )

            rows_base = fit_eval_models(
                data=data,
                features=base_features,
                target=target,
                experiment_type="base_set0",
                feature_set_name="SET0_BASE",
                extra={
                    "horizon": h,
                    "target_threshold_pct": pct,
                    "target_threshold": thr,
                }
            )

            all_rows.extend(rows_baseline)
            all_rows.extend(rows_base)

    out = pd.DataFrame(all_rows)

    out.to_csv(
        OUTPUT_DIR / "1_1_A_baseline_and_set0_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return out


# ============================================================
# 10. ЭТАП B: SET0 + КАЖДЫЙ ДЕТЕРМИНАНТ
# ============================================================

def run_individual_determinants(df):
    print("\n=== LINK 1.1B SET0 + each determinant ===")

    base_features = existing_features(df, SET0_BASE)
    determinants = existing_features(df, ALL_DETERMINANTS)

    missing = [f for f in ALL_DETERMINANTS if f not in df.columns]

    with open(OUTPUT_DIR / "missing_determinants.txt", "w", encoding="utf-8") as f:
        for m in missing:
            f.write(m + "\n")

    all_rows = []
    all_period_rows = []

    total = len(determinants) * len(HORIZONS) * len(TARGET_THRESHOLDS)
    counter = 0

    for feature in determinants:
        for h in HORIZONS:
            for thr in TARGET_THRESHOLDS:
                counter += 1

                pct = int(round(thr * 100))
                target = f"target_direction_{h}d_thr_{pct}pct"

                common_features = base_features + [feature]
                data_common = make_model_data(df, common_features, target)

                if data_common.empty:
                    continue

                train, val, test = split_data(data_common)

                period_row = {
                    "asset": ASSET_NAME,
                    "feature": feature,
                    "feature_group": feature_group_name(feature),
                    "horizon": h,
                    "target_threshold_pct": pct,
                    "target_threshold": thr,
                    "target": target,
                    "data_start": data_common["date"].min(),
                    "data_end": data_common["date"].max(),
                    **split_info(train, val, test, target),
                }

                all_period_rows.append(period_row)

                if not data_ok(train, val, test, target):
                    continue

                rows_baseline_same = fit_eval_models(
                    data=data_common,
                    features=[],
                    target=target,
                    experiment_type="individual_baseline_same_dates",
                    feature_set_name=f"Baseline_same_dates_for_{feature}",
                    extra={
                        "feature": feature,
                        "feature_group": feature_group_name(feature),
                        "horizon": h,
                        "target_threshold_pct": pct,
                        "target_threshold": thr,
                    }
                )

                rows_base_same = fit_eval_models(
                    data=data_common,
                    features=base_features,
                    target=target,
                    experiment_type="individual_set0_same_dates",
                    feature_set_name=f"SET0_same_dates_for_{feature}",
                    extra={
                        "feature": feature,
                        "feature_group": feature_group_name(feature),
                        "horizon": h,
                        "target_threshold_pct": pct,
                        "target_threshold": thr,
                    }
                )

                rows_augmented = fit_eval_models(
                    data=data_common,
                    features=common_features,
                    target=target,
                    experiment_type="individual_set0_plus_feature",
                    feature_set_name=f"SET0_PLUS_{feature}",
                    extra={
                        "feature": feature,
                        "feature_group": feature_group_name(feature),
                        "horizon": h,
                        "target_threshold_pct": pct,
                        "target_threshold": thr,
                    }
                )

                all_rows.extend(rows_baseline_same)
                all_rows.extend(rows_base_same)
                all_rows.extend(rows_augmented)

                if counter % 50 == 0:
                    print(f"{counter}/{total}: feature={feature}, h={h}, thr={pct}%")

    results = pd.DataFrame(all_rows)
    periods = pd.DataFrame(all_period_rows)

    results.to_csv(
        OUTPUT_DIR / "1_1_B_individual_raw_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    periods.to_csv(
        OUTPUT_DIR / "1_1_B_feature_periods_by_horizon_threshold.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return results, periods


# ============================================================
# 11. DELTA: SET0 + FEATURE МИНУС SET0
# ============================================================

def make_individual_deltas(raw_results):
    if raw_results.empty:
        return pd.DataFrame()

    base = raw_results[
        raw_results["experiment_type"] == "individual_set0_same_dates"
    ].copy()

    aug = raw_results[
        raw_results["experiment_type"] == "individual_set0_plus_feature"
    ].copy()

    merge_cols = [
        "feature",
        "feature_group",
        "horizon",
        "target_threshold_pct",
        "target_threshold",
        "target",
        "model",
        "eval_split",
    ]

    base_cols = merge_cols + [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "train_class_1_share",
        "validation_class_1_share",
        "test_class_1_share",
    ]

    merged = aug.merge(
        base[base_cols],
        on=merge_cols,
        how="left",
        suffixes=("", "_set0_same_dates")
    )

    for metric in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        merged[f"delta_{metric}"] = (
            merged[metric] - merged[f"{metric}_set0_same_dates"]
        )

    merged.to_csv(
        OUTPUT_DIR / "1_1_C_individual_deltas_vs_set0.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return merged


# ============================================================
# 12. РЕКОМЕНДАЦИИ ПО ДЕТЕРМИНАНТАМ
# ============================================================

def make_feature_recommendations(deltas, periods):
    if deltas.empty:
        return pd.DataFrame()

    selection = deltas[
        (deltas["eval_split"] == "validation") &
        (deltas["model"].isin(SELECTION_MODELS))
    ].copy()

    if selection.empty:
        return pd.DataFrame()

    rows = []

    for feature, g in selection.groupby("feature"):
        feature_group = g["feature_group"].iloc[0]

        avg_delta_auc = g["delta_roc_auc"].mean()
        median_delta_auc = g["delta_roc_auc"].median()

        avg_delta_bal = g["delta_balanced_accuracy"].mean()
        median_delta_bal = g["delta_balanced_accuracy"].median()

        avg_delta_acc = g["delta_accuracy"].mean()
        avg_delta_f1 = g["delta_f1"].mean()

        positive_auc_rate = (g["delta_roc_auc"] > 0).mean()
        positive_bal_rate = (g["delta_balanced_accuracy"] > 0).mean()

        n_tests = len(g)
        n_auc_ge_005 = (g["delta_roc_auc"] >= 0.005).sum()
        n_bal_ge_005 = (g["delta_balanced_accuracy"] >= 0.005).sum()

        min_train_rows = g["train_rows"].min()
        min_val_rows = g["validation_rows"].min()
        min_test_rows = g["test_rows"].min()

        p = periods[periods["feature"] == feature].copy()

        if not p.empty:
            data_start_min = p["data_start"].min()
            data_start_max = p["data_start"].max()
            data_end_min = p["data_end"].min()
            data_end_max = p["data_end"].max()

            train_rows_min = p["train_rows"].min()
            train_rows_max = p["train_rows"].max()
            val_rows_min = p["validation_rows"].min()
            val_rows_max = p["validation_rows"].max()
            test_rows_min = p["test_rows"].min()
            test_rows_max = p["test_rows"].max()
        else:
            data_start_min = data_start_max = data_end_min = data_end_max = pd.NaT
            train_rows_min = train_rows_max = np.nan
            val_rows_min = val_rows_max = np.nan
            test_rows_min = test_rows_max = np.nan

        if min_val_rows < MIN_VAL_ROWS or min_train_rows < MIN_TRAIN_ROWS:
            recommendation = "WEAK_REVIEW_LOW_ROWS"
        elif (
            avg_delta_auc >= 0.005 and
            avg_delta_bal >= 0.002 and
            positive_auc_rate >= 0.55
        ):
            recommendation = "INCLUDE"
        elif (
            avg_delta_auc > 0 or
            avg_delta_bal > 0 or
            positive_auc_rate >= 0.45
        ):
            recommendation = "WEAK_REVIEW"
        else:
            recommendation = "EXCLUDE"

        if recommendation == "INCLUDE":
            explanation = (
                "Признак в среднем улучшает validation-качество относительно SET0 "
                "и может быть включён в следующий этап."
            )
        elif recommendation == "WEAK_REVIEW":
            explanation = (
                "Признак нестабилен: иногда помогает, иногда ухудшает. "
                "Не удалять окончательно; проверить в группах и на отдельных режимах."
            )
        elif recommendation == "WEAK_REVIEW_LOW_ROWS":
            explanation = (
                "Недостаточно наблюдений для уверенного вывода. "
                "Использовать осторожно."
            )
        else:
            explanation = (
                "Признак в среднем не улучшает validation-качество относительно SET0. "
                "Кандидат на исключение, если нет сильного экономического смысла."
            )

        rows.append({
            "asset": ASSET_NAME,
            "feature": feature,
            "feature_group": feature_group,

            "recommendation": recommendation,
            "explanation": explanation,

            "avg_delta_roc_auc_validation": avg_delta_auc,
            "median_delta_roc_auc_validation": median_delta_auc,

            "avg_delta_balanced_accuracy_validation": avg_delta_bal,
            "median_delta_balanced_accuracy_validation": median_delta_bal,

            "avg_delta_accuracy_validation": avg_delta_acc,
            "avg_delta_f1_validation": avg_delta_f1,

            "positive_auc_rate_validation": positive_auc_rate,
            "positive_balanced_accuracy_rate_validation": positive_bal_rate,

            "n_validation_tests": n_tests,
            "n_delta_auc_ge_0_005": int(n_auc_ge_005),
            "n_delta_balanced_accuracy_ge_0_005": int(n_bal_ge_005),

            "min_train_rows_used": min_train_rows,
            "min_validation_rows_used": min_val_rows,
            "min_test_rows_used": min_test_rows,

            "data_start_min": data_start_min,
            "data_start_max": data_start_max,
            "data_end_min": data_end_min,
            "data_end_max": data_end_max,

            "train_rows_min_all_targets": train_rows_min,
            "train_rows_max_all_targets": train_rows_max,

            "validation_rows_min_all_targets": val_rows_min,
            "validation_rows_max_all_targets": val_rows_max,

            "test_rows_min_all_targets": test_rows_min,
            "test_rows_max_all_targets": test_rows_max,
        })

    out = pd.DataFrame(rows)

    order = {
        "INCLUDE": 0,
        "WEAK_REVIEW": 1,
        "WEAK_REVIEW_LOW_ROWS": 2,
        "EXCLUDE": 3,
    }

    out["recommendation_order"] = out["recommendation"].map(order)
    out = out.sort_values(
        [
            "recommendation_order",
            "avg_delta_roc_auc_validation",
            "avg_delta_balanced_accuracy_validation",
        ],
        ascending=[True, False, False]
    ).drop(columns=["recommendation_order"])

    out.to_csv(
        OUTPUT_DIR / "1_1_D_feature_recommendations.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return out


# ============================================================
# 13. ГРАФИКИ
# ============================================================

def plot_top_features(recommendations):
    if recommendations.empty:
        return

    data = recommendations.copy()
    data = data.sort_values("avg_delta_roc_auc_validation", ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    plt.barh(
        data["feature"][::-1],
        data["avg_delta_roc_auc_validation"][::-1],
    )
    plt.axvline(0, linewidth=1)
    plt.title(f"{ASSET_NAME}: Top determinants by average validation delta ROC-AUC")
    plt.xlabel("Avg delta ROC-AUC vs SET0")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top20_determinants_delta_roc_auc.png", dpi=160)
    plt.close()

    data2 = recommendations.copy()
    data2 = data2.sort_values("avg_delta_balanced_accuracy_validation", ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    plt.barh(
        data2["feature"][::-1],
        data2["avg_delta_balanced_accuracy_validation"][::-1],
    )
    plt.axvline(0, linewidth=1)
    plt.title(f"{ASSET_NAME}: Top determinants by average validation delta balanced accuracy")
    plt.xlabel("Avg delta balanced accuracy vs SET0")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top20_determinants_delta_balanced_accuracy.png", dpi=160)
    plt.close()

    counts = recommendations["recommendation"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)
    plt.title(f"{ASSET_NAME}: Feature recommendation counts")
    plt.ylabel("Number of determinants")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recommendation_counts.png", dpi=160)
    plt.close()


def plot_each_feature(deltas, recommendations):
    if deltas.empty or recommendations.empty:
        return

    val = deltas[
        (deltas["eval_split"] == "validation") &
        (deltas["model"].isin(SELECTION_MODELS))
    ].copy()

    if val.empty:
        return

    for feature in sorted(val["feature"].dropna().unique()):
        sub = val[val["feature"] == feature].copy()

        if sub.empty:
            continue

        by_h = (
            sub.groupby("horizon")[["delta_roc_auc", "delta_balanced_accuracy"]]
            .mean()
            .reset_index()
            .sort_values("horizon")
        )

        rec_row = recommendations[recommendations["feature"] == feature]

        if not rec_row.empty:
            rec = rec_row["recommendation"].iloc[0]
            group = rec_row["feature_group"].iloc[0]
            avg_auc = rec_row["avg_delta_roc_auc_validation"].iloc[0]
            avg_bal = rec_row["avg_delta_balanced_accuracy_validation"].iloc[0]
        else:
            rec = "NA"
            group = "NA"
            avg_auc = np.nan
            avg_bal = np.nan

        plt.figure(figsize=(10, 5))

        plt.plot(
            by_h["horizon"],
            by_h["delta_roc_auc"],
            marker="o",
            label="Delta ROC-AUC"
        )

        plt.plot(
            by_h["horizon"],
            by_h["delta_balanced_accuracy"],
            marker="o",
            label="Delta balanced accuracy"
        )

        plt.axhline(0, linewidth=1)
        plt.title(
            f"{ASSET_NAME}: {feature} | group={group} | rec={rec}\n"
            f"avg delta AUC={avg_auc:.4f}, avg delta bal_acc={avg_bal:.4f}"
        )
        plt.xlabel("Forecast horizon, days")
        plt.ylabel("Average validation delta vs SET0")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fname = safe_filename(feature) + ".png"
        plt.savefig(FEATURE_PLOTS_DIR / fname, dpi=160)
        plt.close()


# ============================================================
# 14. TXT-ОТЧЁТ
# ============================================================

def write_summary(base_results, deltas, recommendations, periods):
    path = OUTPUT_DIR / "1_1_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SECTION 1.1 — {ASSET_NAME}: BASELINE, SET0, SET0 + EACH DETERMINANT\n")
        f.write("=====================================================\n\n")

        f.write(f"Asset: {ASSET_NAME}\n")
        f.write(f"Input file: {FILE_PATH}\n")
        f.write(f"Sheet: {SHEET_NAME}\n")
        f.write(f"Horizons: {HORIZONS}\n")
        f.write(f"Target thresholds: {TARGET_THRESHOLDS}\n")
        f.write(f"Train: {TRAIN_START} — {TRAIN_END}\n")
        f.write(f"Validation: {VAL_START} — {VAL_END}\n")
        f.write(f"Test: {TEST_START} — {TEST_END}\n\n")

        f.write("Logic:\n")
        f.write("1. Baseline = majority class model.\n")
        f.write("2. SET0 = LINK own lags and rolling volatility.\n")
        f.write("3. Each determinant tested as SET0 + one feature.\n")
        f.write("4. SET0 and SET0+feature are compared on exactly the same dates.\n")
        f.write("5. Feature recommendation is based only on validation results.\n\n")

        if not recommendations.empty:
            f.write("Recommendation counts:\n")
            counts = recommendations["recommendation"].value_counts()
            for rec, cnt in counts.items():
                f.write(f"- {rec}: {cnt}\n")

            f.write("\nTop 15 by avg validation delta ROC-AUC:\n")
            top = recommendations.sort_values(
                "avg_delta_roc_auc_validation",
                ascending=False
            ).head(15)

            for _, row in top.iterrows():
                f.write(
                    f"- {row['feature']} | group={row['feature_group']} | "
                    f"rec={row['recommendation']} | "
                    f"avg_delta_auc={row['avg_delta_roc_auc_validation']:.5f} | "
                    f"avg_delta_bal_acc={row['avg_delta_balanced_accuracy_validation']:.5f} | "
                    f"period_start_range={row['data_start_min']} — {row['data_start_max']} | "
                    f"period_end_range={row['data_end_min']} — {row['data_end_max']}\n"
                )

        f.write("\nFiles created:\n")
        f.write("- 1_1_A_baseline_and_set0_results.csv\n")
        f.write("- 1_1_B_individual_raw_results.csv\n")
        f.write("- 1_1_B_feature_periods_by_horizon_threshold.csv\n")
        f.write("- 1_1_C_individual_deltas_vs_set0.csv\n")
        f.write("- 1_1_D_feature_recommendations.csv\n")
        f.write("- 1_1_report.xlsx\n")
        f.write("- plots/top20_determinants_delta_roc_auc.png\n")
        f.write("- plots/top20_determinants_delta_balanced_accuracy.png\n")
        f.write("- plots/features/*.png\n")


# ============================================================
# 15. EXCEL-ОТЧЁТ
# ============================================================

def write_excel_report(base_results, raw_results, periods, deltas, recommendations):
    xlsx_path = OUTPUT_DIR / "1_1_report.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if not base_results.empty:
            base_results.to_excel(writer, sheet_name="baseline_set0", index=False)

        if not recommendations.empty:
            recommendations.to_excel(writer, sheet_name="feature_recommendations", index=False)

        if not deltas.empty:
            deltas.to_excel(writer, sheet_name="feature_deltas", index=False)

        if not periods.empty:
            periods.to_excel(writer, sheet_name="feature_periods", index=False)

        if not raw_results.empty:
            raw_results.to_excel(writer, sheet_name="raw_individual_results", index=False)


# ============================================================
# 16. MAIN
# ============================================================

def main():
    ensure_dirs()

    print(f"Starting Section 1.1 pipeline for {ASSET_NAME}...")
    print(f"Output folder: {OUTPUT_DIR}")

    df = load_data()

    print("\nConfigured base features:")
    print(existing_features(df, SET0_BASE))

    print("\nConfigured determinants:")
    for group, feats in FEATURE_GROUPS.items():
        ok = existing_features(df, feats)
        missing = [f for f in feats if f not in df.columns]
        print(f"\n{group}")
        print("found:", ok)
        print("missing:", missing)

    base_results = run_baseline_and_base(df)

    raw_results, periods = run_individual_determinants(df)

    deltas = make_individual_deltas(raw_results)

    recommendations = make_feature_recommendations(deltas, periods)

    print("\nSaving plots...")
    plot_top_features(recommendations)
    plot_each_feature(deltas, recommendations)

    print("\nWriting reports...")
    write_summary(base_results, deltas, recommendations, periods)
    write_excel_report(base_results, raw_results, periods, deltas, recommendations)

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_DIR}")

    print("\nMain files:")
    print("- 1_1_summary.txt")
    print("- 1_1_report.xlsx")
    print("- 1_1_A_baseline_and_set0_results.csv")
    print("- 1_1_B_individual_raw_results.csv")
    print("- 1_1_B_feature_periods_by_horizon_threshold.csv")
    print("- 1_1_C_individual_deltas_vs_set0.csv")
    print("- 1_1_D_feature_recommendations.csv")
    print("- plots/top20_determinants_delta_roc_auc.png")
    print("- plots/top20_determinants_delta_balanced_accuracy.png")
    print("- plots/features/")


if __name__ == "__main__":
    main()