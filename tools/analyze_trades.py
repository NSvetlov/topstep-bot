import sys
import pandas as pd


def summarize(path: str) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print("no rows")
        return

    # Basic stats
    total = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] < 0).sum()
    evens = (df["pnl"] == 0).sum()
    net = df["pnl"].sum()
    avg = df["pnl"].mean()
    best = df["pnl"].max()
    worst = df["pnl"].min()

    print(f"total={total} wins={wins} losses={losses} breakeven={evens}")
    print(f"net={net:.2f} avg={avg:.2f} best={best:.2f} worst={worst:.2f}")

    # Duration
    try:
        t_in = pd.to_datetime(df["entry_time"])
        t_out = pd.to_datetime(df["exit_time"])
        dur_min = (t_out - t_in).dt.total_seconds() / 60.0
        print(f"median_dur_min={dur_min.median():.1f} mean_dur_min={dur_min.mean():.1f}")
        df["dur_min"] = dur_min
    except Exception:
        pass

    if "mode" in df.columns:
        print("-- by mode --")
        g = df.groupby("mode")["pnl"].agg(["count", "sum", "mean", "median"]).sort_index()
        for idx, row in g.iterrows():
            print(f"mode={idx} n={int(row['count'])} sum={row['sum']:.2f} avg={row['mean']:.2f} med={row['median']:.2f}")

    if "reason" in df.columns:
        print("-- by reason --")
        g = df.groupby("reason")["pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
        for idx, row in g.iterrows():
            print(f"reason={idx} n={int(row['count'])} sum={row['sum']:.2f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mnq_trades.csv"
    summarize(path)
