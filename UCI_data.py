from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import io
import urllib.request
import zipfile

import pandas as pd

# Optional deps (works without them if you don't call those datasets)
try:
    import openml  # type: ignore
except Exception:
    openml = None  # type: ignore

try:
    from ucimlrepo import fetch_ucirepo  # type: ignore
except Exception:
    fetch_ucirepo = None  # type: ignore


@dataclass(frozen=True)
class DatasetSpec:
    # If present, try UCI "new" API first
    uci_id: Optional[int] = None
    uci_name: Optional[str] = None

    # Fallback: direct URL to a file (csv/txt/xls/xlsx/zip)
    url: Optional[str] = None
    reader: Optional[Callable[[str], pd.DataFrame]] = None

    # Default target column name (scalar regression)
    default_target: Optional[str] = None


def _read_csv(url: str, **kw) -> pd.DataFrame:
    return pd.read_csv(url, **kw)


def _read_excel(url: str, **kw) -> pd.DataFrame:
    return pd.read_excel(url, **kw)


def _read_uci_zip_member(
    url: str,
    member_suffix: str,
    *,
    sep=r"\s+",
    header=None,
) -> pd.DataFrame:
    """Download a ZIP from `url`, find a member that endswith `member_suffix`, read it as a table."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        matches = [m for m in z.namelist() if m.endswith(member_suffix)]
        if not matches:
            preview = z.namelist()[:30]
            raise FileNotFoundError(
                f"ZIP from {url} has no member ending with {member_suffix}. First members: {preview}"
            )
        member = matches[0]
        with z.open(member) as f:
            return pd.read_csv(f, sep=sep, header=header, engine="python")


def _ensure_single_col_df(y: object, name_hint: str = "y") -> pd.DataFrame:
    """Convert Series/scalar-array/DF to a 1-col DataFrame; error if DF has >1 col."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Expected scalar target, got y with {y.shape[1]} columns: {list(y.columns)}")
        return y
    if isinstance(y, pd.Series):
        return y.to_frame()
    # last resort: try to DataFrame-ify; ensure 1 col
    y_df = pd.DataFrame(y)
    if y_df.shape[1] != 1:
        raise ValueError(f"Expected scalar target, got {name_hint} with shape {y_df.shape}")
    return y_df


def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not in columns: {list(df.columns)}")
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


SPECS: dict[str, DatasetSpec] = {
    # --- UCI direct (older) ---
    "boston": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        reader=lambda u: _read_csv(u, sep=r'\s+', header=None),
        default_target="MEDV",
    ),
    "wine": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        reader=lambda u: _read_csv(u, header=None),
        default_target="class",  # note: classification-ish; kept for compatibility
    ),

    # --- UCI direct (known file locations) ---
    "concrete": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        reader=lambda u: _read_excel(u),
        default_target="ConcreteCompressiveStrength",
    ),
    "energy": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        reader=lambda u: _read_excel(u),
        default_target="Y1",  # scalar default; overrideable to Y2
    ),
    "yacht": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        reader=lambda u: _read_csv(u, sep=r'\s+', header=None),
        default_target="residuary_resistance",
    ),
    "protein": DatasetSpec(
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
        reader=lambda u: _read_csv(u),
        default_target="RMSD",
    ),

    # --- UCI "new" catalog (use ucimlrepo) + fallbacks ---
    "power": DatasetSpec(
        uci_id=294,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
        reader=lambda u: _read_uci_zip_member(u, "Folds5x2_pp.xlsx", header=0),
        default_target="PE",
    ),
    "naval": DatasetSpec(
        uci_id=316,
        url="https://archive.ics.uci.edu/static/public/316/condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip",
        reader=lambda u: _read_uci_zip_member(u, "data.txt", sep=r"\s+", header=None),
        default_target="kMc",  # scalar default; overrideable to kMt
    ),

    # --- OpenML ---
    "kin8nm": DatasetSpec(default_target=None),  # OpenML id=189 has a default target attribute
}


def load_dataset(key: str, *, target: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X, y) as DataFrames, where y is always a *single-column* DataFrame.

    key in: boston, concrete, energy, kin8nm, naval, power, protein, wine, yacht
    target: optional override for the scalar target column (e.g. energy: 'Y2', naval: 'kMt')
    """
    key = key.strip().lower()
    if key not in SPECS:
        raise KeyError(f"Unknown key '{key}'. Options: {sorted(SPECS)}")

    spec = SPECS[key]

    # --- OpenML special-case ---
    if key == "kin8nm":
        if openml is None:
            raise ImportError("pip install openml")
        ds = openml.datasets.get_dataset(189)
        X, y, *_ = ds.get_data(target=ds.default_target_attribute)
        y_df = _ensure_single_col_df(y, "openml y")
        # if user wants a different target on OpenML, they'd need a different dataset;
        # we keep API consistent and ignore `target` here (or you can raise if target is not None).
        if target is not None:
            raise ValueError("OpenML kin8nm exposes one default target; `target=` override is not supported here.")
        return X, y_df

    # 1) Try ucimlrepo (if installed) — but enforce scalar y
    if fetch_ucirepo is not None and (spec.uci_id is not None or spec.uci_name is not None):
        try:
            dset = fetch_ucirepo(id=spec.uci_id) if spec.uci_id is not None else fetch_ucirepo(name=spec.uci_name)
            X = dset.data.features
            y = dset.data.targets

            # Choose scalar target from ucimlrepo output
            y_df = y if isinstance(y, pd.DataFrame) else pd.DataFrame(y)

            chosen = target or spec.default_target
            if chosen is not None:
                if chosen in y_df.columns:
                    return X, y_df[[chosen]]
                # sometimes targets come unnamed; handle 1-col case
                if y_df.shape[1] == 1:
                    return X, _ensure_single_col_df(y_df, "uci y")
                raise KeyError(f"Requested target '{chosen}' not in ucimlrepo targets: {list(y_df.columns)}")

            # no chosen name; accept only if scalar
            return X, _ensure_single_col_df(y_df, "uci y")

        except Exception as e:
            # print(f"[load_dataset:{key}] ucimlrepo fetch failed: {type(e).__name__}: {e}")
            pass

    # 2) Direct download fallback
    if not spec.url or not spec.reader:
        raise RuntimeError(f"No direct fallback configured for '{key}'")

    df = spec.reader(spec.url)

    # ---- Dataset-specific cleanup / column naming ----
    if key == "boston":
        df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",
                      "PTRATIO", "B", "LSTAT", "MEDV"]

    elif key == "wine":
        df.columns = ["class", "Alcohol", "Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium",
                      "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
                      "Color_intensity", "Hue", "OD280_OD315", "Proline"]

    elif key == "yacht":
        df.columns = ["LCB", "Prismatic_coeff", "Length_displacement_ratio", "Beam_draught_ratio",
                      "Length_beam_ratio", "Froude_number", "residuary_resistance"]

    elif key == "concrete":
        # If headers missing, set them:
        if df.columns.tolist() == list(range(len(df.columns))):
            df.columns = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                          "CoarseAggregate", "FineAggregate", "Age", "ConcreteCompressiveStrength"]

    elif key == "energy":
        # Often includes an unnamed first column; drop it if it looks like an index
        if df.columns.size > 10 and str(df.columns[0]).lower().startswith("unnamed"):
            df = df.drop(columns=[df.columns[0]])
        # If still unlabeled, do best-effort label the 10 known fields
        if df.shape[1] >= 10 and all(isinstance(c, int) for c in df.columns[:10]):
            df = df.iloc[:, :10].copy()
            df.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]

    elif key == "naval":
        # 16 features + 2 targets
        if df.shape[1] == 18 and all(isinstance(c, int) for c in df.columns):
            df.columns = [
                "lp", "v", "GTT", "GTn", "GGn", "Ts", "Tp", "T48", "T1", "T2",
                "P48", "P1", "P2", "Pexh", "TIC", "mf",
                "kMc", "kMt",
            ]

    elif key == "power":
        # If excel read yields unnamed columns, leave them; we'll split by target name or last column
        pass

    # ---- Scalar target selection ----
    chosen_target = target or spec.default_target

    if chosen_target is not None and chosen_target in df.columns:
        X, y = _split_xy(df, chosen_target)
        return X, y

    # If chosen_target was requested but not found, error early (better than silently picking last col)
    if target is not None:
        raise KeyError(f"Requested target '{target}' not found for dataset '{key}'. Columns: {list(df.columns)}")

    # Otherwise: best-effort scalar split
    # - If spec.default_target exists but wasn't found, fall back to last column.
    # - If dataset has multiple targets (energy/naval), default_target should exist by design above.
    y = df.iloc[:, [-1]]
    X = df.iloc[:, :-1]
    return X, y


