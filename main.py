"""
=============================================================================
EDS_TUPM-25-0519_Ganal
Engineering Data Systems Pipeline
HVA-02: Air Handling Unit (AHU) Static Pressure Analysis
Student: Ian Marlo S. Ganal | TUPM-25-0519
Course: Computer Programming | AY 2026
=============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH   = "data/dataset_original.csv"
CLEANED_PATH   = "data/dataset_cleaned.csv"
OUTPUT_DIR     = "outputs"
UNIQUE_FILTER  = "2020"          # Ian's unique filter: Year 2020 only

os.makedirs("data",    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# MODULE 1 — DATA INGESTION
# =============================================================================
class DataIngestion:
    """Loads the raw HVAC BMS dataset from CSV and applies a unique filter."""

    def __init__(self, path: str, unique_filter: str):
        self.path          = path
        self.unique_filter = unique_filter
        self.raw_df        = None

    def load(self) -> pd.DataFrame:
        print("=" * 65)
        print("  MODULE 1 — DATA INGESTION")
        print("=" * 65)
        try:
            df = pd.read_csv(self.path)
            print(f"  [OK] Loaded '{self.path}'  →  {df.shape[0]:,} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print(f"  [ERROR] File not found: {self.path}")
            sys.exit(1)
        except Exception as e:
            print(f"  [ERROR] Could not read file: {e}")
            sys.exit(1)

        # Parse timestamp
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            df["Timestamp"] = df["Timestamp"].dt.tz_convert("Europe/Berlin")
        except Exception as e:
            print(f"  [WARN] Timestamp parsing issue: {e}")

        # ── Unique Filter: Year 2020 ──────────────────────────────────────
        try:
            before = len(df)
            df = df[df["Timestamp"].dt.year == int(self.unique_filter)]
            print(f"  [FILTER] Year={self.unique_filter} → {len(df):,} / {before:,} rows retained")
        except Exception as e:
            print(f"  [WARN] Filter could not be applied: {e}")

        self.raw_df = df.copy()
        print(f"  Date range : {df['Timestamp'].min()} → {df['Timestamp'].max()}")
        print()
        return df


# =============================================================================
# MODULE 2 — DATA CLEANING
# =============================================================================
class DataCleaning:
    """Handles null values, duplicates, type correction, and outlier capping."""

    NUMERIC_COLS = [
        "T_Supply", "T_Return", "SP_Return",
        "T_Saturation", "T_Outdoor",
        "RH_Supply", "RH_Return", "RH_Outdoor",
        "Energy", "Power"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean(self) -> pd.DataFrame:
        print("=" * 65)
        print("  MODULE 2 — DATA CLEANING")
        print("=" * 65)
        df = self.df

        # Step 1 — Missing values
        missing_before = df.isnull().sum().sum()
        try:
            for col in self.NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df[self.NUMERIC_COLS] = df[self.NUMERIC_COLS].fillna(
                df[self.NUMERIC_COLS].median()
            )
            missing_after = df.isnull().sum().sum()
            print(f"  [OK] Missing values: {missing_before} → {missing_after}")
        except Exception as e:
            print(f"  [WARN] Missing value handling error: {e}")

        # Step 2 — Duplicates
        try:
            dups = df.duplicated().sum()
            df = df.drop_duplicates()
            print(f"  [OK] Duplicate rows removed: {dups}")
        except Exception as e:
            print(f"  [WARN] Duplicate removal error: {e}")

        # Step 3 — Type correction
        try:
            df["Energy"] = df["Energy"].astype(float)
            df["Power"]  = df["Power"].astype(float)
            print(f"  [OK] Data types corrected")
        except Exception as e:
            print(f"  [WARN] Type correction error: {e}")

        # Step 4 — Outlier capping (IQR method)
        try:
            capped = 0
            for col in ["T_Supply", "T_Return", "RH_Supply", "RH_Return", "Power"]:
                if col not in df.columns:
                    continue
                Q1  = df[col].quantile(0.25)
                Q3  = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - 3 * IQR, Q3 + 3 * IQR
                n = ((df[col] < lo) | (df[col] > hi)).sum()
                df[col] = df[col].clip(lower=lo, upper=hi)
                capped += n
            print(f"  [OK] Outlier values capped (IQR ×3): {capped} values adjusted")
        except Exception as e:
            print(f"  [WARN] Outlier capping error: {e}")

        # Step 5 — Feature engineering
        try:
            df["Month"]        = df["Timestamp"].dt.month
            df["Hour"]         = df["Timestamp"].dt.hour
            df["Season"]       = df["Month"].map(
                {10:"Autumn",11:"Autumn",12:"Winter",
                  1:"Winter", 2:"Winter", 3:"Spring",
                  4:"Spring", 5:"Spring", 6:"Summer",
                  7:"Summer", 8:"Summer", 9:"Autumn"}
            )
            df["T_Delta"]      = df["T_Supply"] - df["T_Return"]
            df["RH_Delta"]     = df["RH_Supply"] - df["RH_Return"]
            print(f"  [OK] Feature engineering: Month, Hour, Season, T_Delta, RH_Delta added")
        except Exception as e:
            print(f"  [WARN] Feature engineering error: {e}")

        print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print()

        # Save cleaned dataset
        try:
            df.to_csv(CLEANED_PATH, index=False)
            print(f"  [SAVED] Cleaned dataset → '{CLEANED_PATH}'")
        except Exception as e:
            print(f"  [WARN] Could not save cleaned file: {e}")

        print()
        return df


# =============================================================================
# MODULE 3 — STATISTICAL ANALYSIS
# =============================================================================
class StatisticalAnalysis:
    """
    Computes descriptive statistics, distribution analysis,
    correlation, and comparative group analysis using NumPy.
    """

    TARGET_COLS = ["T_Supply", "T_Return", "SP_Return",
                   "RH_Supply", "RH_Return", "Power"]

    def __init__(self, df: pd.DataFrame):
        self.df      = df
        self.results = {}

    # ── Helper ────────────────────────────────────────────────────────────
    def _numpy_stats(self, arr: np.ndarray) -> dict:
        arr = arr[~np.isnan(arr)]
        return {
            "mean"     : np.mean(arr),
            "median"   : np.median(arr),
            "std"      : np.std(arr, ddof=1),
            "variance" : np.var(arr, ddof=1),
            "min"      : np.min(arr),
            "max"      : np.max(arr),
            "skewness" : float(stats.skew(arr)),
            "kurtosis" : float(stats.kurtosis(arr)),
        }

    # ── Descriptive ───────────────────────────────────────────────────────
    def descriptive(self):
        print("=" * 65)
        print("  MODULE 3A — DESCRIPTIVE STATISTICS (NumPy)")
        print("=" * 65)
        rows = []
        for col in self.TARGET_COLS:
            if col not in self.df.columns:
                continue
            s = self._numpy_stats(self.df[col].to_numpy(dtype=float))
            s["variable"] = col
            rows.append(s)
            print(f"\n  [{col}]")
            print(f"    Mean     = {s['mean']:.4f}")
            print(f"    Median   = {s['median']:.4f}")
            print(f"    Std Dev  = {s['std']:.4f}")
            print(f"    Variance = {s['variance']:.4f}")
            print(f"    Skewness = {s['skewness']:.4f}")
            print(f"    Min/Max  = {s['min']:.4f} / {s['max']:.4f}")

        self.results["descriptive"] = pd.DataFrame(rows).set_index("variable")
        print()

    # ── Distribution ──────────────────────────────────────────────────────
    def distribution(self):
        print("=" * 65)
        print("  MODULE 3B — DISTRIBUTION ANALYSIS")
        print("=" * 65)
        for col in ["T_Supply", "SP_Return", "RH_Supply"]:
            if col not in self.df.columns:
                continue
            arr  = self.df[col].dropna().to_numpy(dtype=float)
            skew = stats.skew(arr)
            kurt = stats.kurtosis(arr)

            direction = "right-skewed (tail towards higher values)" if skew > 0.5 \
                else "left-skewed (tail towards lower values)" if skew < -0.5 \
                else "approximately symmetric"
            print(f"\n  [{col}]")
            print(f"    Skewness = {skew:.4f}  →  {direction}")
            print(f"    Kurtosis = {kurt:.4f}")

            # Z-score outlier count
            z     = np.abs(stats.zscore(arr))
            n_out = int(np.sum(z > 3))
            print(f"    Outliers (|Z|>3) = {n_out}")
        print()

    # ── Correlation ───────────────────────────────────────────────────────
    def correlation(self):
        print("=" * 65)
        print("  MODULE 3C — CORRELATION ANALYSIS")
        print("=" * 65)
        cols = [c for c in self.TARGET_COLS if c in self.df.columns]
        corr_matrix = np.corrcoef(
            self.df[cols].dropna().to_numpy(dtype=float).T
        )
        self.results["correlation"] = pd.DataFrame(
            corr_matrix, index=cols, columns=cols
        )
        # Key pairs
        pairs = [
            ("T_Supply",  "SP_Return"),
            ("T_Supply",  "RH_Supply"),
            ("T_Return",  "RH_Return"),
            ("Power",     "T_Supply"),
        ]
        for a, b in pairs:
            if a in cols and b in cols:
                i, j = cols.index(a), cols.index(b)
                r    = corr_matrix[i, j]
                strength = "strong" if abs(r) > 0.7 else \
                           "moderate" if abs(r) > 0.4 else "weak"
                direction = "positive" if r > 0 else "negative"
                print(f"  {a} ↔ {b}: r = {r:.4f}  ({strength} {direction})")
        print()

    # ── Comparative ───────────────────────────────────────────────────────
    def comparative(self):
        print("=" * 65)
        print("  MODULE 3D — COMPARATIVE ANALYSIS (Season Groups)")
        print("=" * 65)
        if "Season" not in self.df.columns:
            print("  [SKIP] Season column not found.")
            return

        season_stats = {}
        for season, grp in self.df.groupby("Season"):
            arr = grp["T_Supply"].dropna().to_numpy(dtype=float)
            season_stats[season] = {
                "n"      : len(arr),
                "mean"   : np.mean(arr),
                "std"    : np.std(arr, ddof=1),
                "median" : np.median(arr),
            }
        self.results["seasonal"] = pd.DataFrame(season_stats).T

        print(f"\n  {'Season':<10} {'N':>6} {'Mean T_Supply':>14} {'Std':>8} {'Median':>8}")
        print("  " + "-" * 52)
        for s, v in season_stats.items():
            print(f"  {s:<10} {v['n']:>6,} {v['mean']:>14.4f} {v['std']:>8.4f} {v['median']:>8.4f}")

        # Welch t-test: Winter vs Summer
        w = self.df[self.df["Season"] == "Winter"]["T_Supply"].dropna().to_numpy(float)
        s = self.df[self.df["Season"] == "Summer"]["T_Supply"].dropna().to_numpy(float)
        if len(w) > 1 and len(s) > 1:
            t_stat, p_val = stats.ttest_ind(w, s, equal_var=False)
            sig = "significant" if p_val < 0.05 else "not significant"
            print(f"\n  Welch t-test (Winter vs Summer T_Supply):")
            print(f"    t = {t_stat:.4f},  p = {p_val:.6f}  →  {sig} (α=0.05)")
        print()

    def run_all(self):
        self.descriptive()
        self.distribution()
        self.correlation()
        self.comparative()
        return self.results


# =============================================================================
# MODULE 4 — STATIC VISUALIZATIONS
# =============================================================================
class StaticVisualizer:
    """Generates 3 static publication-quality charts."""

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df  = df
        self.out = output_dir

    # ── Plot 1: Histogram — T_Supply distribution ─────────────────────────
    def plot_histogram(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("AHU Supply & Return Air Temperature Distribution (2020)",
                     fontsize=13, fontweight="bold")

        for ax, col, color, label in zip(
            axes,
            ["T_Supply", "T_Return"],
            ["#2196F3", "#FF5722"],
            ["Supply Air Temperature (°C)", "Return Air Temperature (°C)"]
        ):
            arr = self.df[col].dropna().to_numpy(float)
            ax.hist(arr, bins=50, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(np.mean(arr),   color="black",  linestyle="--",
                       linewidth=1.5, label=f"Mean={np.mean(arr):.2f}")
            ax.axvline(np.median(arr), color="orange", linestyle=":",
                       linewidth=1.5, label=f"Median={np.median(arr):.2f}")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.out, "plot1_histogram.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")

    # ── Plot 2: Boxplot — Seasonal T_Supply comparison ────────────────────
    def plot_boxplot(self):
        if "Season" not in self.df.columns:
            return
        order   = ["Winter", "Spring", "Summer", "Autumn"]
        present = [s for s in order if s in self.df["Season"].values]
        data    = [self.df[self.df["Season"] == s]["T_Supply"].dropna().values
                   for s in present]
        colors  = ["#5C85D6", "#66BB6A", "#FFA726", "#EF5350"]

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], colors[:len(present)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticklabels(present, fontsize=11)
        ax.set_xlabel("Season", fontsize=11)
        ax.set_ylabel("Supply Air Temperature (°C)", fontsize=11)
        ax.set_title("Seasonal Comparison of AHU Supply Air Temperature (2020)",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.35)

        plt.tight_layout()
        path = os.path.join(self.out, "plot2_boxplot.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")

    # ── Plot 3: Correlation Heatmap ───────────────────────────────────────
    def plot_heatmap(self, corr_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 7))
        cols = corr_df.columns.tolist()
        mat  = corr_df.to_numpy(dtype=float)

        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)

        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center",
                        va="center", fontsize=8,
                        color="black" if abs(mat[i, j]) < 0.7 else "white")

        ax.set_title("Pearson Correlation Heatmap — HVAC BMS Variables (2020)",
                     fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        path = os.path.join(self.out, "plot3_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")

    # ── Plot 4: Scatter — SP_Return vs T_Supply ───────────────────────────
    def plot_scatter(self):
        fig, ax = plt.subplots(figsize=(9, 6))
        sc = ax.scatter(
            self.df["T_Supply"], self.df["SP_Return"],
            c=self.df["RH_Supply"], cmap="coolwarm",
            alpha=0.3, s=5
        )
        plt.colorbar(sc, ax=ax, label="RH Supply (%)")

        # Regression line
        x = self.df["T_Supply"].dropna().to_numpy(float)
        y = self.df["SP_Return"].dropna().to_numpy(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        m, b, r, *_ = stats.linregress(x[mask], y[mask])
        xr = np.linspace(x.min(), x.max(), 200)
        ax.plot(xr, m * xr + b, "k--", linewidth=1.5,
                label=f"Regression: y={m:.3f}x+{b:.2f}  r={r:.3f}")

        ax.set_xlabel("Supply Air Temperature T_Supply (°C)", fontsize=11)
        ax.set_ylabel("Return Air Setpoint SP_Return (°C)",   fontsize=11)
        ax.set_title("T_Supply vs SP_Return — Coloured by RH_Supply (2020)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.out, "plot4_scatter.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {path}")

    def run_all(self, corr_df: pd.DataFrame):
        print("=" * 65)
        print("  MODULE 4 — STATIC VISUALIZATIONS")
        print("=" * 65)
        self.plot_histogram()
        self.plot_boxplot()
        self.plot_heatmap(corr_df)
        self.plot_scatter()
        print()


# =============================================================================
# MODULE 5 — ANIMATED VISUALIZATIONS
# =============================================================================
class AnimatedVisualizer:
    """Produces 2 animated plots (Matplotlib Animation)."""

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df  = df
        self.out = output_dir

    # ── Animation 1: Monthly rolling mean of T_Supply ─────────────────────
    def animate_monthly_trend(self):
        df = self.df.copy()
        df = df.set_index("Timestamp").resample("D")["T_Supply"].mean().reset_index()
        df = df.dropna()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_xlim(df["Timestamp"].min(), df["Timestamp"].max())
        ax.set_ylim(df["T_Supply"].min() - 1, df["T_Supply"].max() + 1)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Mean Daily T_Supply (°C)", fontsize=10)
        ax.set_title("AHU Supply Air Temperature — Daily Mean Trend (2020)",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

        line, = ax.plot([], [], color="#2196F3", linewidth=1.5)
        dot,  = ax.plot([], [], "ro", markersize=5)
        text  = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                        fontsize=9, color="darkred")

        timestamps = df["Timestamp"].values
        values     = df["T_Supply"].values

        def update(frame):
            line.set_data(timestamps[:frame], values[:frame])
            if frame > 0:
                dot.set_data([timestamps[frame - 1]], [values[frame - 1]])
                text.set_text(f"Date: {pd.Timestamp(timestamps[frame-1]).strftime('%Y-%m-%d')}  "
                              f"T={values[frame-1]:.2f}°C")
            return line, dot, text

        frames = len(timestamps)
        step   = max(1, frames // 120)   # cap at ~120 frames for speed
        ani = animation.FuncAnimation(
            fig, update, frames=range(1, frames, step),
            interval=60, blit=True
        )

        path = os.path.join(self.out, "anim1_supply_temp_trend.gif")
        try:
            ani.save(path, writer="pillow", fps=15, dpi=100)
            print(f"  [SAVED] {path}")
        except Exception as e:
            print(f"  [WARN] Could not save animation 1: {e}")
        plt.close()

    # ── Animation 2: Hourly RH_Supply distribution shift by season ────────
    def animate_rh_distribution(self):
        if "Season" not in self.df.columns:
            return
        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        present = [s for s in seasons if s in self.df["Season"].values]
        colors  = {"Winter":"#5C85D6","Spring":"#66BB6A",
                   "Summer":"#FFA726","Autumn":"#EF5350"}

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(30, 100)
        ax.set_xlabel("RH Supply (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("RH_Supply Distribution Shift Across Seasons (2020)",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        season_text = ax.text(0.72, 0.90, "", transform=ax.transAxes,
                              fontsize=13, fontweight="bold")

        def update(frame):
            ax.cla()
            ax.set_xlim(30, 100)
            ax.set_xlabel("RH Supply (%)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title("RH_Supply Distribution Shift Across Seasons (2020)",
                         fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)

            # show progressive reveal
            for i, s in enumerate(present):
                if i > frame:
                    break
                arr = self.df[self.df["Season"] == s]["RH_Supply"].dropna().values
                alpha = 0.85 if i == frame else 0.35
                ax.hist(arr, bins=40, color=colors.get(s, "gray"),
                        alpha=alpha, label=s, edgecolor="white")

            ax.legend(fontsize=9, loc="upper left")
            season_text = ax.text(0.72, 0.90, present[frame],
                                  transform=ax.transAxes, fontsize=13,
                                  fontweight="bold", color=colors.get(present[frame], "gray"))
            return []

        ani = animation.FuncAnimation(
            fig, update, frames=len(present),
            interval=1200, repeat=True
        )
        path = os.path.join(self.out, "anim2_rh_distribution_seasons.gif")
        try:
            ani.save(path, writer="pillow", fps=1, dpi=100)
            print(f"  [SAVED] {path}")
        except Exception as e:
            print(f"  [WARN] Could not save animation 2: {e}")
        plt.close()

    def run_all(self):
        print("=" * 65)
        print("  MODULE 5 — ANIMATED VISUALIZATIONS")
        print("=" * 65)
        self.animate_monthly_trend()
        self.animate_rh_distribution()
        print()


# =============================================================================
# MAIN — PIPELINE ORCHESTRATOR
# =============================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Engineering Data Systems Pipeline — HVA-02                ║")
    print("║   AHU Static Pressure & BMS Analytics                       ║")
    print("║   Ian Marlo S. Ganal | TUPM-25-0519                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Copy raw dataset into data/ folder ───────────────────────────────
    raw_src = "HVAC_NE_EC_19-21.csv"
    if os.path.exists(raw_src) and not os.path.exists(DATASET_PATH):
        import shutil
        shutil.copy(raw_src, DATASET_PATH)

    # ── Module 1: Ingest ─────────────────────────────────────────────────
    ingestor = DataIngestion(DATASET_PATH, UNIQUE_FILTER)
    raw_df   = ingestor.load()

    # ── Module 2: Clean ──────────────────────────────────────────────────
    cleaner  = DataCleaning(raw_df)
    clean_df = cleaner.clean()

    # ── Module 3: Analyze ────────────────────────────────────────────────
    analyzer = StatisticalAnalysis(clean_df)
    results  = analyzer.run_all()

    # ── Module 4: Static Plots ───────────────────────────────────────────
    sv = StaticVisualizer(clean_df, OUTPUT_DIR)
    sv.run_all(results.get("correlation",
               pd.DataFrame()))

    # ── Module 5: Animations ─────────────────────────────────────────────
    av = AnimatedVisualizer(clean_df, OUTPUT_DIR)
    av.run_all()

    print("=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Cleaned data : {CLEANED_PATH}")
    print(f"  Outputs      : {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
