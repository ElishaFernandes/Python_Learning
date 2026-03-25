import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Tuple

from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# 1.  PRICE ORACLE  — trained from the real Nat_Gas CSV
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/Elisha/Desktop/Python learning/JPMorgan/Nat_Gas(2).csv"

def _date_to_features(d: date) -> np.ndarray:
    """
    Convert a date into seasonal Fourier features + a linear time trend.
    This is the same approach from Task 1:
      - t          : days since first date (captures long-run trend)
      - sin/cos 1  : 12-month seasonal cycle
      - sin/cos 2  : 6-month seasonal cycle (harmonic)
    """
    t = (d - date(2000, 1, 1)).days          # anchor point — consistent across calls
    angle_year  = 2 * np.pi * t / 365.25
    angle_half  = 2 * np.pi * t / (365.25 / 2)
    return np.array([
        t,
        np.sin(angle_year), np.cos(angle_year),
        np.sin(angle_half), np.cos(angle_half),
    ])


def _build_price_model(filepath: str):
    """
    Load the CSV, parse dates and prices, fit a seasonal linear regression,
    and return (model, scaler_t0) ready for prediction.
    """
    df = pd.read_csv(filepath)

    # Normalise column names — handle spaces, capitalisation
    df.columns = df.columns.str.strip().str.lower()

    # Identify date and price columns flexibly
    date_col  = next(c for c in df.columns if "date" in c)
    price_col = next(c for c in df.columns if any(k in c for k in ["price", "nat", "gas", "value"]))

    df[date_col]  = pd.to_datetime(df[date_col], dayfirst=False)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])

    dates  = df[date_col].dt.date.tolist()
    prices = df[price_col].values

    X = np.array([_date_to_features(d) for d in dates])
    y = prices

    model = LinearRegression().fit(X, y)
    print(f"  Price model trained on {len(dates)} data points  "
          f"(R² = {model.score(X, y):.4f})")
    return model


# Build the model once at import time
print("  Loading gas price data and training price oracle...")
_PRICE_MODEL = _build_price_model(DATA_PATH)


def get_price(query_date: date) -> float:
    """
    Return the estimated natural gas price on any given date.
    Uses the seasonal regression model trained on Nat_Gas(2).csv.
    """
    features = _date_to_features(query_date).reshape(1, -1)
    price    = float(_PRICE_MODEL.predict(features)[0])
    return round(max(price, 0.0), 4)   # prices can't be negative


# ---------------------------------------------------------------------------
# 2.  PHYSICAL CONSTRAINT HELPERS
# ---------------------------------------------------------------------------

def _days_between(d1: date, d2: date) -> int:
    return (d2 - d1).days


def _volume_injected(
    injection_date: date,
    injection_rate: float,       # MMBtu per day
    max_volume: float,           # MMBtu
    current_volume: float,
) -> float:
    """
    Volume actually injected on a single injection date.
    Capped by (max_volume - current_volume) and the daily rate.
    """
    space_available = max_volume - current_volume
    return min(injection_rate, space_available)


def _volume_withdrawn(
    withdrawal_date: date,
    withdrawal_rate: float,      # MMBtu per day
    current_volume: float,
) -> float:
    """
    Volume actually withdrawn on a single withdrawal date.
    Capped by current_volume and the daily rate.
    """
    return min(withdrawal_rate, current_volume)


# ---------------------------------------------------------------------------
# 3.  CORE PRICING FUNCTION
# ---------------------------------------------------------------------------

def price_storage_contract(
    injection_dates:    List[date],
    withdrawal_dates:   List[date],
    injection_rate:     float,        # MMBtu/day maximum rate
    withdrawal_rate:    float,        # MMBtu/day maximum rate
    max_volume:         float,        # MMBtu maximum storage capacity
    storage_cost_per_day: float,      # $ per day while gas is in store
    injection_cost_per_mmbtu:   float = 0.0,  # $ per MMBtu injected
    withdrawal_cost_per_mmbtu:  float = 0.0,  # $ per MMBtu withdrawn
    transport_cost_per_event:   float = 0.0,  # $ flat fee per inj/with event
    verbose: bool = True,
) -> float:
    """
    Price a natural gas storage contract.

    Parameters
    ----------
    injection_dates            : sorted list of dates on which gas is bought & stored
    withdrawal_dates           : sorted list of dates on which gas is sold
    injection_rate             : max MMBtu that can be pushed in per day
    withdrawal_rate            : max MMBtu that can be pulled out per day
    max_volume                 : storage capacity in MMBtu
    storage_cost_per_day       : daily holding cost in $ (charged for every day
                                 gas remains in storage)
    injection_cost_per_mmbtu   : variable cost charged per MMBtu injected
    withdrawal_cost_per_mmbtu  : variable cost charged per MMBtu withdrawn
    transport_cost_per_event   : flat transport cost per injection or withdrawal event
    verbose                    : print a cash-flow breakdown if True

    Returns
    -------
    contract_value : float   (positive = profitable for buyer)
    """

    # ── Validate inputs ────────────────────────────────────────────────────
    if not injection_dates:
        raise ValueError("At least one injection date is required.")
    if not withdrawal_dates:
        raise ValueError("At least one withdrawal date is required.")
    if max(injection_dates) >= min(withdrawal_dates):
        raise ValueError(
            "All injections must occur strictly before all withdrawals."
        )

    # Sort dates (defensive)
    injection_dates  = sorted(injection_dates)
    withdrawal_dates = sorted(withdrawal_dates)

    # ── Accumulators ───────────────────────────────────────────────────────
    revenue          = 0.0   # cash IN  from selling gas
    purchase_cost    = 0.0   # cash OUT buying gas
    inj_cost_total   = 0.0
    with_cost_total  = 0.0
    transport_total  = 0.0
    storage_total    = 0.0

    current_volume   = 0.0
    log_rows         = []

    # ── INJECTION EVENTS ───────────────────────────────────────────────────
    for inj_date in injection_dates:
        buy_price = get_price(inj_date)
        vol       = _volume_injected(inj_date, injection_rate, max_volume, current_volume)

        cost_buy      = vol * buy_price
        cost_inj      = vol * injection_cost_per_mmbtu
        cost_transport= transport_cost_per_event

        purchase_cost   += cost_buy
        inj_cost_total  += cost_inj
        transport_total += cost_transport
        current_volume  += vol

        log_rows.append({
            "date": inj_date, "event": "INJECT",
            "price": buy_price, "volume": vol,
            "cash_flow": -(cost_buy + cost_inj + cost_transport),
            "storage_volume_after": current_volume,
        })

    # ── STORAGE COST  (charged for every day gas is in store) ──────────────
    # Simplified: cost accrues from first injection to last withdrawal
    total_days = _days_between(injection_dates[0], withdrawal_dates[-1])
    # More accurate: weight by average volume held each day
    # Here we use the average of (volume after last injection, volume before first withdrawal)
    avg_volume      = current_volume  # gas is in store this whole period
    storage_total   = storage_cost_per_day * total_days
    # NOTE: a more precise model would integrate volume * cost_per_mmbtu_per_day
    # over time, but storage contracts usually quote a flat daily fee.

    # ── WITHDRAWAL EVENTS ──────────────────────────────────────────────────
    for with_date in withdrawal_dates:
        sell_price = get_price(with_date)
        vol        = _volume_withdrawn(with_date, withdrawal_rate, current_volume)

        rev_sell       = vol * sell_price
        cost_with      = vol * withdrawal_cost_per_mmbtu
        cost_transport = transport_cost_per_event

        revenue         += rev_sell
        with_cost_total += cost_with
        transport_total += cost_transport
        current_volume  -= vol

        log_rows.append({
            "date": with_date, "event": "WITHDRAW",
            "price": sell_price, "volume": vol,
            "cash_flow": rev_sell - cost_with - cost_transport,
            "storage_volume_after": current_volume,
        })

    # ── CONTRACT VALUE ─────────────────────────────────────────────────────
    total_costs     = purchase_cost + inj_cost_total + with_cost_total + transport_total + storage_total
    contract_value  = revenue - total_costs

    # ── VERBOSE BREAKDOWN ──────────────────────────────────────────────────
    if verbose:
        df = pd.DataFrame(log_rows)
        print("\n" + "="*60)
        print("  STORAGE CONTRACT CASH-FLOW BREAKDOWN")
        print("="*60)
        print(df.to_string(index=False))
        print("-"*60)
        print(f"  Revenue from sales          : ${revenue:>12,.2f}")
        print(f"  Gas purchase cost           : ${-purchase_cost:>12,.2f}")
        print(f"  Injection cost              : ${-inj_cost_total:>12,.2f}")
        print(f"  Withdrawal cost             : ${-with_cost_total:>12,.2f}")
        print(f"  Transport cost (total)      : ${-transport_total:>12,.2f}")
        print(f"  Storage cost ({total_days} days)     : ${-storage_total:>12,.2f}")
        print("="*60)
        print(f"  CONTRACT VALUE              : ${contract_value:>12,.2f}")
        print("="*60 + "\n")

    return contract_value


# ---------------------------------------------------------------------------
# 4.  SCENARIO TESTS
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Scenario A: Simple summer-buy / winter-sell ────────────────────────
    print("\n>>> SCENARIO A: Classic summer buy / winter sell")
    val_a = price_storage_contract(
        injection_dates   = [date(2024, 6, 1), date(2024, 7, 1)],
        withdrawal_dates  = [date(2024, 12, 1), date(2025, 1, 1)],
        injection_rate    = 1_000_000,    # 1M MMBtu/day
        withdrawal_rate   = 1_000_000,
        max_volume        = 5_000_000,    # 5M MMBtu capacity
        storage_cost_per_day      = 2_500,   # $2,500/day
        injection_cost_per_mmbtu  = 0.01,
        withdrawal_cost_per_mmbtu = 0.01,
        transport_cost_per_event  = 50_000,
    )

    # ── Scenario B: Unprofitable contract (costs exceed spread) ───────────
    print("\n>>> SCENARIO B: High-cost contract (expect negative value)")
    val_b = price_storage_contract(
        injection_dates   = [date(2024, 5, 1)],
        withdrawal_dates  = [date(2024, 11, 1)],
        injection_rate    = 500_000,
        withdrawal_rate   = 500_000,
        max_volume        = 1_000_000,
        storage_cost_per_day      = 50_000,   # extremely high storage fee
        injection_cost_per_mmbtu  = 0.50,
        withdrawal_cost_per_mmbtu = 0.50,
        transport_cost_per_event  = 200_000,
    )

    # ── Scenario C: Multi-injection, multi-withdrawal ─────────────────────
    print("\n>>> SCENARIO C: Multiple injection and withdrawal dates")
    val_c = price_storage_contract(
        injection_dates   = [date(2024, 4, 1), date(2024, 5, 15), date(2024, 6, 30)],
        withdrawal_dates  = [date(2024, 11, 1), date(2024, 12, 15), date(2025, 1, 31)],
        injection_rate    = 800_000,
        withdrawal_rate   = 800_000,
        max_volume        = 3_000_000,
        storage_cost_per_day      = 3_000,
        injection_cost_per_mmbtu  = 0.005,
        withdrawal_cost_per_mmbtu = 0.005,
        transport_cost_per_event  = 30_000,
    )

    print(f"\nSummary of contract values:")
    print(f"  Scenario A : ${val_a:>12,.2f}")
    print(f"  Scenario B : ${val_b:>12,.2f}")
    print(f"  Scenario C : ${val_c:>12,.2f}")