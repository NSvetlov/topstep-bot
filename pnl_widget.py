import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from typing import List, Tuple, Optional, Dict


class PnLWidget:
    """
    Lightweight Matplotlib popup that plots equity over time
    and displays rolling statistics (realized/unrealized, max DD, wins/losses).

    Use in non-blocking mode inside a long-running loop by calling
    `update(ts, equity, realized, unrealized, stats)` periodically.
    """

    def __init__(self, title: str = "PnL", figsize=(8, 4)) -> None:
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title("Equity Curve")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Equity ($)")
        self.ax.grid(True, alpha=0.3)

        # equity time series
        self._times: List[datetime] = []
        self._equity: List[float] = []
        (self._line,) = self.ax.plot([], [], label="Equity", color="#1f77b4")
        self.ax.legend(loc="upper left")

        # text box for stats
        self._text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        self.fig.tight_layout()
        self.fig.show()

    def update(
        self,
        ts: datetime,
        equity: float,
        realized: float,
        unrealized: float,
        stats: Optional[Dict[str, float]] = None,
    ) -> None:
        # append new point
        self._times.append(ts)
        self._equity.append(float(equity))

        # keep last N to avoid memory bloat
        if len(self._times) > 5000:
            self._times = self._times[-4000:]
            self._equity = self._equity[-4000:]

        # update line
        self._line.set_data(self._times, self._equity)
        if self._times:
            self.ax.set_xlim(self._times[0], self._times[-1])
            ymin = min(self._equity)
            ymax = max(self._equity)
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            pad = max(5.0, 0.05 * (ymax - ymin))
            self.ax.set_ylim(ymin - pad, ymax + pad)

        # update stats text
        s = stats or {}
        wins = int(s.get("wins", 0))
        losses = int(s.get("losses", 0))
        trades = wins + losses
        win_rate = (100.0 * wins / trades) if trades > 0 else 0.0
        max_dd = float(s.get("max_dd", 0.0))
        text = (
            f"Realized: {realized:,.2f}\n"
            f"Unrealized: {unrealized:,.2f}\n"
            f"Equity: {equity:,.2f}\n"
            f"Max DD: {max_dd:,.2f}\n"
            f"Trades: {trades}  Win%: {win_rate:.1f}%"
        )
        self._text.set_text(text)

        # refresh without blocking main loop
        self.fig.canvas.draw_idle()
        try:
            # brief pause to process GUI events
            plt.pause(0.001)
        except Exception:
            pass

