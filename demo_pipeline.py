"""
Minimal reproducible demo pipeline.

The full project pipeline is intentionally larger and supports raw-data imports.
This demo starts from a small aligned sensor dataset and keeps the core analysis
path: sliding-window features, shared embedding, KDE atlas regions,
subject/label/region occupancy summaries, and a few representative figures.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, mannwhitneyu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
RESULTS_DIR = SCRIPT_DIR / "Results"

METADATA_COLUMNS = {"Time", "Marker", "Group", "sheet", "GroundTruth"}
FIXED_MARKER_ORDER = ["Stand", "Sit", "Sit2Stand", "Walking", "Step", "Turn", "Squat"]
FIXED_MARKER_COLORS = {
    "Stand": "#1f77b4",
    "Sit": "#ff7f0e",
    "Sit2Stand": "#2ca02c",
    "Walking": "#d62728",
    "Step": "#9467bd",
    "Turn": "#8c564b",
    "Squat": "#e377c2",
    "Unknown": "#7f7f7f",
}
GROUP_COLORS_SCATTER = {"Control": "#1f77b4", "Injured": "#d62728", "Unknown": "#7f7f7f"}
GROUP_COLORS = {"Control": "#4c78a8", "Injured": "#f58518", "Unknown": "#7f7f7f"}
GROUP_ORDER = ["Control", "Injured"]
FAMILY_COLORS = {"stable": "#3a7ca5", "complex": "#d95f02", "other": "#7570b3"}
RAW_P_NOTE = "Raw Mann-Whitney U p-values: * p<0.05; ~ p<0.10 trend. Demo data is intentionally small."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Unsupervised-Human-Motion-Analysis demo pipeline.")
    parser.add_argument(
        "--subject",
        nargs="*",
        help="Optional subject IDs, e.g. S1 or S1 S13. Default: all demo subjects.",
    )
    parser.add_argument("--window", type=int, default=200, help="Sliding window length in samples.")
    parser.add_argument("--step", type=int, default=50, help="Sliding window step in samples.")
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne", help="2D embedding method.")
    parser.add_argument("--perplexity", type=float, default=18.0, help="t-SNE perplexity.")
    parser.add_argument("--grid", type=int, default=240, help="KDE grid resolution.")
    parser.add_argument("--bandwidth", type=float, default=0.25, help="KDE bandwidth.")
    parser.add_argument("--levels", type=int, default=18, help="Contour levels for heatmaps.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma for atlas segmentation.")
    parser.add_argument("--min_distance", type=int, default=12, help="Minimum peak distance in KDE pixels.")
    parser.add_argument("--threshold_rel", type=float, default=0.08, help="Relative peak threshold.")
    parser.add_argument("--min_region", type=int, default=25, help="Minimum segmented region size in pixels.")
    parser.add_argument("--smooth_windows", type=int, default=5, help="Majority smoothing window over region IDs.")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def clean_label(value) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return "Unknown"
    return text


def sheet_sort_key(name: str) -> int:
    text = str(name).strip().upper()
    if text.startswith("S") and text[1:].isdigit():
        return int(text[1:])
    return 10**9


def activity_sort_key(label: str) -> tuple[int, str]:
    if label in FIXED_MARKER_ORDER:
        return FIXED_MARKER_ORDER.index(label), label
    return len(FIXED_MARKER_ORDER), label


def family_of_label(label: str) -> str:
    if label in {"Stand", "Step", "Walking"}:
        return "stable"
    if label in {"Turn", "Squat", "Sit2Stand"}:
        return "complex"
    return "other"


def load_manifest(subjects: list[str] | None) -> pd.DataFrame:
    manifest_path = DATA_DIR / "demo_subjects.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing demo data manifest: {manifest_path}. Run prepare_demo_data.py once from scripts_DEMO."
        )

    manifest = pd.read_csv(manifest_path)
    required = {"sheet", "group", "file"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Demo manifest missing columns: {sorted(missing)}")

    manifest["sheet"] = manifest["sheet"].astype(str).str.upper()
    if subjects:
        requested = [s.strip().upper() for s in subjects if s.strip()]
        missing_subjects = sorted(set(requested).difference(set(manifest["sheet"])), key=sheet_sort_key)
        if missing_subjects:
            raise RuntimeError(f"Requested subject(s) not found in demo data: {missing_subjects}")
        manifest = manifest[manifest["sheet"].isin(requested)].copy()

    manifest = manifest.sort_values("sheet", key=lambda s: s.map(sheet_sort_key)).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("No demo subjects selected.")
    return manifest


def output_dir_for(manifest: pd.DataFrame) -> Path:
    if len(manifest) == 1:
        return RESULTS_DIR / str(manifest.iloc[0]["sheet"])
    return RESULTS_DIR / "cohort_shared"


def load_subject_signal(row: pd.Series) -> pd.DataFrame:
    path = DATA_DIR / str(row["file"])
    if not path.exists():
        raise FileNotFoundError(f"Missing demo signal file: {path}")

    df = pd.read_csv(path)
    required = ["Time", "Marker", "Group"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Group"] = str(row["group"])
    df["sheet"] = str(row["sheet"])
    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    sensor_cols = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            df[col] = numeric.interpolate(limit_direction="both")
            sensor_cols.append(col)

    if not sensor_cols:
        raise ValueError(f"{path} does not contain any numeric sensor columns.")

    df = df[["Time", "Marker", "Group", "sheet"] + sensor_cols].copy()
    df.attrs["sensor_columns"] = sensor_cols
    return df


def feature_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0])

    diffs = np.diff(values)
    x = np.arange(values.size, dtype=float)
    if values.size > 1 and np.std(x) > 0:
        slope = float(np.polyfit(x, values, 1)[0])
    else:
        slope = 0.0

    q25, q75 = np.percentile(values, [25, 75])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "iqr": float(q75 - q25),
        "range": float(np.max(values) - np.min(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "energy": float(np.mean(values**2)),
        "mean_abs_diff": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
        "slope": slope,
    }


def extract_window_features(df: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    marker = (
        df["Marker"]
        .replace(r"^\s*$", np.nan, regex=True)
        .ffill()
        .fillna("Unknown")
        .map(clean_label)
        .to_numpy(dtype=object)
    )
    group = df["Group"].replace(r"^\s*$", np.nan, regex=True).ffill().fillna("Unknown").map(clean_label)
    sensor_cols = [col for col in df.columns if col not in METADATA_COLUMNS]
    if not sensor_cols:
        raise RuntimeError("No numeric sensor columns available for feature extraction.")
    signal_df = df[sensor_cols].copy()

    rows = []
    for start in range(0, len(df) - window + 1, step):
        end = start + window
        center = start + window // 2
        row = {
            "Time": float(df["Time"].iloc[center]),
            "GroundTruth": clean_label(marker[center]),
            "Group": clean_label(group.iloc[center]),
            "sheet": str(df["sheet"].iloc[0]),
        }
        for channel in sensor_cols:
            stats = feature_stats(signal_df[channel].iloc[start:end].to_numpy(dtype=float))
            for name, value in stats.items():
                row[f"{channel}__{name}"] = value
        rows.append(row)

    if not rows:
        raise RuntimeError(f"Not enough rows ({len(df)}) for window={window}, step={step}.")
    return pd.DataFrame(rows)


def build_feature_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    metadata = {"Time", "GroundTruth", "Group", "sheet"}
    feature_cols = [col for col in features.columns if col not in metadata]
    X = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True)).fillna(0.0)
    X_scaled = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return X_scaled, feature_cols


def run_embedding(features: pd.DataFrame, method: str, perplexity: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, feature_cols = build_feature_matrix(features)
    n_components = min(15, X.shape[0] - 1, X.shape[1])
    if n_components < 2:
        raise RuntimeError("Not enough windows/features for a 2D embedding.")

    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(X)
    pca_df = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    if method == "pca" or len(Xp) < 8:
        Y = Xp[:, :2]
    else:
        tsne_perplexity = min(float(perplexity), max(2.0, (len(Xp) - 1) / 3.0))
        tsne_perplexity = min(tsne_perplexity, len(Xp) - 1.0)
        try:
            tsne = TSNE(
                n_components=2,
                perplexity=tsne_perplexity,
                learning_rate=200,
                max_iter=1000,
                init="pca",
                random_state=random_state,
                verbose=0,
            )
        except TypeError:
            tsne = TSNE(
                n_components=2,
                perplexity=tsne_perplexity,
                learning_rate=200,
                n_iter=1000,
                init="pca",
                random_state=random_state,
                verbose=0,
            )
        Y = tsne.fit_transform(Xp)

    out = pd.DataFrame(
        {
            "Time": features["Time"].to_numpy(),
            "x": Y[:, 0],
            "y": Y[:, 1],
            "GroundTruth": features["GroundTruth"].map(clean_label).to_numpy(),
            "Group": features["Group"].map(clean_label).to_numpy(),
            "sheet": features["sheet"].astype(str).to_numpy(),
        }
    )
    out.attrs["feature_columns"] = feature_cols
    return out, pca_df


def save_embedding_plots(embed_df: pd.DataFrame, out_dir: Path, method: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for group in ["Control", "Injured", "Unknown"]:
        sub = embed_df[embed_df["Group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=16,
            alpha=0.72,
            color=GROUP_COLORS_SCATTER.get(group, "#7f7f7f"),
            edgecolors="none",
            label=group,
        )
    ax.set_title(f"Shared Step3 Embedding by Group ({method.upper()})")
    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    fig.savefig(out_dir / "step3_shared_embedding_by_group.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    labels = sorted(pd.unique(embed_df["GroundTruth"]), key=activity_sort_key)
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    for label in labels:
        sub = embed_df[embed_df["GroundTruth"] == label]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=13,
            alpha=0.75,
            color=FIXED_MARKER_COLORS.get(label, "#7f7f7f"),
            edgecolors="none",
            label=label,
        )
    ax.set_title(f"Shared Step3 Embedding by Activity ({method.upper()})")
    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(4, max(1, len(labels))))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    fig.savefig(out_dir / "step3_shared_embedding_by_label.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def build_density(embed_df: pd.DataFrame, bandwidth: float, grid: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = embed_df["x"].to_numpy(dtype=float)
    y = embed_df["y"].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max((xmax - xmin) * 0.08, 1e-3)
    ypad = max((ymax - ymin) * 0.08, 1e-3)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    X = np.vstack([x, y])
    try:
        kde = gaussian_kde(X, bw_method=bandwidth)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(random_state)
        kde = gaussian_kde(X + rng.normal(scale=1e-6, size=X.shape), bw_method=bandwidth)

    xs = np.linspace(xmin, xmax, int(grid))
    ys = np.linspace(ymin, ymax, int(grid))
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    coords = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(coords).reshape(xx.shape)
    return xx, yy, xs, ys, normalize_matrix(zz)


def segment_regions(zi_norm: np.ndarray, xs: np.ndarray, ys: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], list[int]]:
    zi_blur = gaussian_filter(zi_norm, sigma=args.sigma)
    peak_coords = peak_local_max(
        zi_blur,
        min_distance=args.min_distance,
        threshold_rel=args.threshold_rel,
    )
    if len(peak_coords) == 0:
        peak_coords = np.array([np.unravel_index(int(np.argmax(zi_blur)), zi_blur.shape)])

    markers = np.zeros_like(zi_blur, dtype=int)
    peak_values = []
    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[int(r), int(c)] = idx
        peak_values.append((idx, float(zi_blur[int(r), int(c)]), int(r), int(c)))

    mask = zi_blur > max(float(args.threshold_rel) * float(np.nanmax(zi_blur)), 1e-8)
    labels = watershed(-zi_blur, markers=markers, mask=mask)

    removed: list[int] = []
    for rid in sorted(int(x) for x in np.unique(labels) if int(x) > 0):
        if int(np.sum(labels == rid)) < int(args.min_region):
            labels[labels == rid] = 0
            removed.append(rid)

    active = sorted(int(x) for x in np.unique(labels) if int(x) > 0)
    relabel = {old: new for new, old in enumerate(active, start=1)}
    relabelled = np.zeros_like(labels, dtype=int)
    for old, new in relabel.items():
        relabelled[labels == old] = new

    tags = []
    peak_lookup = {idx: (val, r, c) for idx, val, r, c in peak_values}
    for old, new in relabel.items():
        val, r, c = peak_lookup.get(old, (np.nan, np.nan, np.nan))
        tags.append({"id": int(new), "x": float(xs[int(c)]), "y": float(ys[int(r)]), "val": float(val)})
    tags.sort(key=lambda row: row["id"])
    return zi_blur, relabelled, tags, removed


def assign_points(embed_df: pd.DataFrame, xs: np.ndarray, ys: np.ndarray, labels: np.ndarray, smooth_windows: int) -> pd.DataFrame:
    rows = embed_df.copy()
    xi = np.searchsorted(xs, rows["x"].to_numpy(dtype=float), side="left")
    yi = np.searchsorted(ys, rows["y"].to_numpy(dtype=float), side="left")
    xi = np.clip(xi, 0, len(xs) - 1)
    yi = np.clip(yi, 0, len(ys) - 1)
    rows["region_id_raw"] = labels[yi, xi].astype(int)

    parts = []
    win = max(1, int(smooth_windows))
    if win % 2 == 0:
        win += 1
    half = win // 2
    for _, sub in rows.groupby("sheet", sort=False):
        sub = sub.sort_values("Time").copy()
        raw = sub["region_id_raw"].astype(int).to_numpy()
        smooth = np.empty_like(raw)
        for i in range(len(raw)):
            lo = max(0, i - half)
            hi = min(len(raw), i + half + 1)
            vals, counts = np.unique(raw[lo:hi], return_counts=True)
            smooth[i] = int(vals[np.argmax(counts)])
        sub["region_id_smooth"] = smooth
        sub["region_id"] = sub["region_id_smooth"]
        parts.append(sub)
    return pd.concat(parts, axis=0, ignore_index=True)


def save_density_plot(xx: np.ndarray, yy: np.ndarray, zi_plot: np.ndarray, out_path: Path, levels: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    contour = ax.contourf(xx, yy, zi_plot, levels=levels, cmap="turbo")
    ax.contour(xx, yy, zi_plot, levels=levels, colors="k", linewidths=0.25, alpha=0.30)
    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.set_title("Shared Behaviour Density")
    fig.colorbar(contour, ax=ax, label="Density (normalized)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_overlay_plot(embed_df: pd.DataFrame, xx: np.ndarray, yy: np.ndarray, zi_plot: np.ndarray, labels: np.ndarray, tags: list[dict], out_path: Path, levels: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    base = ax.contourf(xx, yy, zi_plot, levels=levels, cmap="turbo", alpha=0.85)
    fig.colorbar(base, ax=ax, label="Density (normalized)")
    ax.contour(xx, yy, labels, colors="white", linewidths=0.65, alpha=0.85)

    for group in ["Control", "Injured", "Unknown"]:
        sub = embed_df[embed_df["Group"].astype(str) == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=14,
            alpha=0.55,
            facecolors="none" if group == "Control" else GROUP_COLORS_SCATTER.get(group, "#7f7f7f"),
            edgecolors=GROUP_COLORS_SCATTER.get(group, "#7f7f7f"),
            linewidths=0.45,
            label=group,
            zorder=3,
        )

    for tag in tags:
        ax.text(
            tag["x"],
            tag["y"],
            str(tag["id"]),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.16", fc="black", alpha=0.72),
            zorder=4,
        )

    ax.set_xlabel("Embedding X")
    ax.set_ylabel("Embedding Y")
    ax.set_title("Shared Behaviour Atlas Overlay")
    ax.legend(frameon=False, markerscale=2.0, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_region_stats(points_df: pd.DataFrame, tags: list[dict]) -> pd.DataFrame:
    tag_dict = {int(t["id"]): t for t in tags}
    rows = []
    for rid in sorted(int(x) for x in pd.unique(points_df["region_id"]) if int(x) > 0):
        sub = points_df[points_df["region_id"] == rid]
        gt_vc = sub["GroundTruth"].astype(str).value_counts(dropna=False)
        group_vc = sub["Group"].astype(str).value_counts(dropna=False)
        rows.append(
            {
                "region_id": rid,
                "point_count": int(len(sub)),
                "subject_count": int(sub["sheet"].nunique()),
                "peak_x": float(tag_dict.get(rid, {}).get("x", np.nan)),
                "peak_y": float(tag_dict.get(rid, {}).get("y", np.nan)),
                "peak_density": float(tag_dict.get(rid, {}).get("val", np.nan)),
                "top_label": str(gt_vc.index[0]) if len(gt_vc) else "Unknown",
                "top_label_ratio": float(gt_vc.iloc[0] / gt_vc.sum()) if len(gt_vc) else np.nan,
                "top_group": str(group_vc.index[0]) if len(group_vc) else "Unknown",
                "top_group_ratio": float(group_vc.iloc[0] / group_vc.sum()) if len(group_vc) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_occupancy(points_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = points_df[points_df["region_id"].astype(int) > 0].copy()
    work["region_id"] = work["region_id"].astype(int)
    subject_totals = work.groupby("sheet", as_index=False).size().rename(columns={"size": "total_windows"})
    region_counts = work.groupby(["sheet", "Group", "region_id"], as_index=False).size().rename(columns={"size": "count"})
    region_occ = region_counts.merge(subject_totals, on="sheet", how="left")
    region_occ["fraction"] = region_occ["count"] / region_occ["total_windows"].replace(0, np.nan)

    label_totals = (
        work.groupby(["sheet", "GroundTruth"], as_index=False)
        .size()
        .rename(columns={"size": "label_total_windows"})
    )
    label_region = (
        work.groupby(["sheet", "Group", "GroundTruth", "region_id"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    label_region = label_region.merge(label_totals, on=["sheet", "GroundTruth"], how="left")
    label_region["fraction_within_label"] = label_region["count"] / label_region["label_total_windows"].replace(0, np.nan)
    return region_occ, label_region


def entropy_norm(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(probs.size))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def compute_region_metrics(label_occ: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_occ.empty:
        return pd.DataFrame(), pd.DataFrame()

    labels = sorted(pd.unique(label_occ["GroundTruth"]).tolist(), key=activity_sort_key)
    control_templates: dict[int, np.ndarray] = {}
    for region_id, sub in label_occ[label_occ["Group"] == "Control"].groupby("region_id"):
        counts = sub.groupby("GroundTruth")["count"].sum().reindex(labels, fill_value=0).to_numpy(dtype=float)
        if counts.sum() > 0:
            control_templates[int(region_id)] = counts / counts.sum()

    if not control_templates:
        for region_id, sub in label_occ.groupby("region_id"):
            counts = sub.groupby("GroundTruth")["count"].sum().reindex(labels, fill_value=0).to_numpy(dtype=float)
            if counts.sum() > 0:
                control_templates[int(region_id)] = counts / counts.sum()

    rows = []
    for (sheet, group, region_id), sub in label_occ.groupby(["sheet", "Group", "region_id"], sort=True):
        counts = sub.groupby("GroundTruth")["count"].sum().reindex(labels, fill_value=0).to_numpy(dtype=float)
        total = float(counts.sum())
        if total <= 0:
            continue
        probs = counts / total
        template = control_templates.get(int(region_id), np.ones(len(labels), dtype=float) / len(labels))
        rows.append(
            {
                "sheet": str(sheet),
                "Group": str(group),
                "region_id": int(region_id),
                "region_label": f"R{int(region_id)}",
                "total_windows": int(total),
                "compactness": float(np.max(probs)),
                "entropy_norm": entropy_norm(probs),
                "effective_regions": float(np.exp(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))),
                "control_template_jsd": jsd(probs, template),
            }
        )

    metrics = pd.DataFrame(rows)
    stats = compute_group_stats(metrics)
    return metrics, stats


def compute_group_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    rows = []
    metric_cols = ["compactness", "entropy_norm", "effective_regions", "control_template_jsd"]
    for region_label, sub in metrics.groupby("region_label", sort=True):
        for metric in metric_cols:
            control = pd.to_numeric(sub.loc[sub["Group"] == "Control", metric], errors="coerce").dropna()
            injured = pd.to_numeric(sub.loc[sub["Group"] == "Injured", metric], errors="coerce").dropna()
            row = {
                "scope": "region",
                "region_label": region_label,
                "metric": metric,
                "n_control": int(len(control)),
                "n_injured": int(len(injured)),
                "control_mean": float(control.mean()) if len(control) else np.nan,
                "injured_mean": float(injured.mean()) if len(injured) else np.nan,
                "delta_mean_injured_minus_control": float(injured.mean() - control.mean()) if len(control) and len(injured) else np.nan,
                "p_raw": np.nan,
            }
            if len(control) >= 2 and len(injured) >= 2:
                _, p_raw = mannwhitneyu(injured, control, alternative="two-sided")
                row["p_raw"] = float(p_raw)
            rows.append(row)
    return pd.DataFrame(rows)


def metric_axis_label(metric: str) -> str:
    return {
        "compactness": "Compactness (dominant activity ratio)",
        "entropy_norm": "Normalised activity entropy",
        "effective_regions": "Effective activities",
        "control_template_jsd": "JSD vs control template",
    }.get(metric, metric)


def format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=n/a"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def save_group_region_barplots(metrics: pd.DataFrame, stats: pd.DataFrame, out_path: Path) -> None:
    if metrics.empty:
        return

    metrics_order = ["compactness", "entropy_norm", "effective_regions", "control_template_jsd"]
    region_order = (
        metrics[["region_id", "region_label"]]
        .drop_duplicates()
        .sort_values("region_id")["region_label"]
        .tolist()
    )
    groups = [g for g in GROUP_ORDER if g in set(metrics["Group"])]
    if not groups:
        groups = sorted(pd.unique(metrics["Group"]).tolist())

    fig_w = max(11.5, 1.35 * len(region_order) + 4.5)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 8.7))
    axes = axes.ravel()
    x = np.arange(len(region_order), dtype=float)

    for ax, metric in zip(axes, metrics_order):
        width = 0.34 if len(groups) > 1 else 0.50
        offsets = np.linspace(-width / 2, width / 2, len(groups)) if len(groups) > 1 else np.array([0.0])
        y_max_seen = 0.0

        for group, offset in zip(groups, offsets):
            means = []
            for region in region_order:
                vals = pd.to_numeric(
                    metrics.loc[(metrics["region_label"] == region) & (metrics["Group"] == group), metric],
                    errors="coerce",
                ).dropna()
                means.append(float(vals.mean()) if len(vals) else np.nan)
                if len(vals):
                    jitter = np.linspace(-0.035, 0.035, len(vals)) if len(vals) > 1 else np.array([0.0])
                    ax.scatter(
                        np.full(len(vals), x[region_order.index(region)] + offset) + jitter,
                        vals,
                        s=18,
                        color=GROUP_COLORS.get(group, "#7f7f7f"),
                        edgecolors="white",
                        linewidths=0.4,
                        alpha=0.90,
                        zorder=3,
                    )
                    y_max_seen = max(y_max_seen, float(vals.max()))
            ax.bar(
                x + offset,
                means,
                width=width,
                color=GROUP_COLORS.get(group, "#7f7f7f"),
                alpha=0.78,
                label=group,
                zorder=2,
            )

        if len(groups) >= 2 and not stats.empty:
            metric_stats = stats[stats["metric"] == metric]
            for pos, region in enumerate(region_order):
                row = metric_stats[metric_stats["region_label"] == region]
                if row.empty:
                    continue
                row = row.iloc[0]
                delta = row.get("delta_mean_injured_minus_control", np.nan)
                p_raw = row.get("p_raw", np.nan)
                if not np.isfinite(delta):
                    continue
                color = "#b2182b" if np.isfinite(p_raw) and p_raw < 0.05 else "#555555"
                ax.text(
                    pos,
                    y_max_seen * 1.08 if y_max_seen > 0 else 1.02,
                    f"d={delta:+.2f}\n{format_p(float(p_raw))}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                )

        ax.set_title(metric_axis_label(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(region_order, rotation=0, ha="center")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if y_max_seen > 0:
            ax.set_ylim(0, y_max_seen * (1.32 if len(groups) >= 2 else 1.15))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.955))
    title = "Injured vs Control by Region ID" if len(groups) >= 2 else "Single-Subject Region Metrics"
    fig.suptitle(title, y=0.99)
    if len(groups) >= 2:
        fig.text(0.5, 0.012, RAW_P_NOTE, ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0.045, 1, 0.925])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def circular_positions(labels: list[str], start_deg: float, end_deg: float, radius: float = 1.0) -> dict[str, tuple[float, float]]:
    if len(labels) == 1:
        angles = np.array([(start_deg + end_deg) / 2.0])
    else:
        angles = np.linspace(start_deg, end_deg, len(labels))
    out = {}
    for label, deg in zip(labels, angles):
        rad = math.radians(float(deg))
        out[label] = (radius * math.cos(rad), radius * math.sin(rad))
    return out


def save_region_activity_network(label_occ: pd.DataFrame, out_path: Path) -> None:
    if label_occ.empty:
        return

    work = label_occ.copy()
    work["GroundTruth"] = work["GroundTruth"].map(clean_label)
    work["region_id"] = pd.to_numeric(work["region_id"], errors="coerce")
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["region_id"]).copy()
    work["region"] = "R" + work["region_id"].astype(int).astype(str)

    region_labels = sorted(pd.unique(work["region"]).tolist(), key=lambda x: int(x[1:]))
    activity_labels = sorted(pd.unique(work["GroundTruth"]).tolist(), key=activity_sort_key)
    links = work.groupby(["region", "GroundTruth"], as_index=False)["count"].sum()
    max_weight = max(float(links["count"].max()), 1.0)

    region_pos = circular_positions(region_labels, 130, 230, radius=1.0)
    activity_pos = circular_positions(activity_labels, -55, 55, radius=1.0)
    pos = {**region_pos, **activity_pos}

    fig, ax = plt.subplots(figsize=(9.0, 8.5))
    ax.set_aspect("equal")
    ax.axis("off")

    for _, row in links.iterrows():
        source = str(row["region"])
        target = clean_label(row["GroundTruth"])
        if source not in pos or target not in pos:
            continue
        weight = float(row["count"])
        x1, y1 = pos[source]
        x2, y2 = pos[target]
        path = MplPath(
            [(x1, y1), (0.0, 0.0), (x2, y2)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor="#6b6b6b",
            linewidth=0.35 + 4.0 * math.sqrt(weight / max_weight),
            alpha=0.14 + 0.36 * (weight / max_weight),
            zorder=1,
        )
        ax.add_patch(patch)

    region_totals = work.groupby("region")["count"].sum().to_dict()
    activity_totals = work.groupby("GroundTruth")["count"].sum().to_dict()
    max_node = max([*region_totals.values(), *activity_totals.values(), 1.0])

    for region in region_labels:
        x, y = pos[region]
        size = 120 + 520 * float(region_totals.get(region, 0.0)) / max_node
        ax.scatter([x], [y], s=size, color="#4d4d4d", edgecolors="white", linewidths=1.0, zorder=4)
        ax.text(x - 0.075, y, region, ha="right", va="center", fontsize=10, color="#222222", weight="bold")

    for label in activity_labels:
        x, y = pos[label]
        family = family_of_label(label)
        size = 120 + 520 * float(activity_totals.get(label, 0.0)) / max_node
        ax.scatter([x], [y], s=size, color=FAMILY_COLORS[family], edgecolors="white", linewidths=1.0, zorder=4)
        ax.text(x + 0.075, y, label, ha="left", va="center", fontsize=10, color="#222222")

    legend_items = []
    for family, color in FAMILY_COLORS.items():
        if any(family_of_label(label) == family for label in activity_labels):
            legend_items.append(
                plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=9, label=f"{family.capitalize()} activity")
            )
    if legend_items:
        ax.legend(handles=legend_items, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(legend_items))

    ax.set_title("Region-Activity Circular Network", fontsize=15, pad=18)
    ax.text(
        0.5,
        0.965,
        f"{len(activity_labels)} activities, {len(region_labels)} regions; edge width = total demo windows",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-1.32, 1.28)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, manifest: pd.DataFrame, features: pd.DataFrame, points: pd.DataFrame, region_stats: pd.DataFrame) -> None:
    group_counts = {str(k): int(v) for k, v in manifest.groupby("group").size().items()}
    lines = [
        "Unsupervised-Human-Motion-Analysis Demo Run Summary",
        "",
        f"Subjects: {', '.join(manifest['sheet'].astype(str).tolist())}",
        f"Groups: {group_counts}",
        f"Feature windows: {len(features)}",
        f"Feature columns: {len([col for col in features.columns if col not in METADATA_COLUMNS])}",
        f"Assigned windows: {len(points)}",
        f"Atlas regions: {len(region_stats)}",
        "",
        "Representative outputs:",
        "  step3_shared_embedding_all.csv",
        "  step4_shared_atlas_heatmap.png",
        "  step4_shared_atlas_overlay.png",
        "  step7_region_activity_circular_network.png",
        "  step7b_group_region_barplots_with_deltas.png",
    ]
    (out_dir / "demo_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.subject)
    out_dir = output_dir_for(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subjects").mkdir(exist_ok=True)

    print(f"[INFO] Demo subjects: {', '.join(manifest['sheet'].tolist())}")
    print(f"[INFO] Output folder: {out_dir}")

    feature_parts = []
    for _, row in manifest.iterrows():
        sheet = str(row["sheet"])
        subject_dir = out_dir / "subjects" / sheet
        subject_dir.mkdir(parents=True, exist_ok=True)

        signals = load_subject_signal(row)
        signals.drop(columns=["sheet"]).to_csv(subject_dir / "step1_demo_signals.csv", index=False)

        features = extract_window_features(signals, window=args.window, step=args.step)
        features.to_csv(subject_dir / "step2_demo_features.csv", index=False)
        feature_parts.append(features)
        sensor_count = len(signals.attrs.get("sensor_columns", []))
        print(f"[OK] {sheet}: {len(signals)} samples, {sensor_count} sensor columns -> {len(features)} windows")

    all_features = pd.concat(feature_parts, axis=0, ignore_index=True)
    all_features.to_csv(out_dir / "step2_demo_features_all.csv", index=False)

    embed_df, pca_df = run_embedding(all_features, args.method, args.perplexity, args.random_state)
    embed_df.to_csv(out_dir / "step3_shared_embedding_all.csv", index=False)
    pca_df.to_csv(out_dir / "step3_shared_pca_variance.csv", index=False)
    save_embedding_plots(embed_df, out_dir, args.method)
    print(f"[OK] Step3 embedding: {len(embed_df)} windows")

    xx, yy, xs, ys, density = build_density(embed_df, args.bandwidth, args.grid, args.random_state)
    density_plot, labels, tags, removed = segment_regions(density, xs, ys, args)
    if removed:
        print(f"[INFO] Removed small atlas regions: {removed}")
    points = assign_points(embed_df, xs, ys, labels, args.smooth_windows)
    region_stats = build_region_stats(points, tags)

    points.to_csv(out_dir / "step4_shared_window_regions.csv", index=False)
    region_stats.to_csv(out_dir / "step4_shared_region_stats.csv", index=False)
    save_density_plot(xx, yy, density_plot, out_dir / "step4_shared_atlas_heatmap.png", args.levels)
    save_overlay_plot(embed_df, xx, yy, density_plot, labels, tags, out_dir / "step4_shared_atlas_overlay.png", args.levels)
    print(f"[OK] Step4 atlas regions: {len(region_stats)}")

    region_occ, label_occ = build_occupancy(points)
    region_occ.to_csv(out_dir / "step5_subject_region_occupancy.csv", index=False)
    label_occ.to_csv(out_dir / "step5_subject_label_region_occupancy.csv", index=False)

    region_metrics, group_stats = compute_region_metrics(label_occ)
    region_metrics.to_csv(out_dir / "step7_subject_region_metrics.csv", index=False)
    group_stats.to_csv(out_dir / "step7_group_region_stats.csv", index=False)

    save_region_activity_network(label_occ, out_dir / "step7_region_activity_circular_network.png")
    barplot_path = out_dir / "step7b_group_region_barplots_with_deltas.png"
    save_group_region_barplots(region_metrics, group_stats, barplot_path)
    if barplot_path.exists():
        shutil.copyfile(barplot_path, out_dir / "step7_group_region_barplots_with_deltas.png")

    write_summary(out_dir, manifest, all_features, points, region_stats)
    print("[DONE] Demo pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
