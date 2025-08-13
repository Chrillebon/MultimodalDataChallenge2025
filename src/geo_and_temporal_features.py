# featureize_meta.py
import math
import numpy as np
import pandas as pd
from pyproj import Transformer
from typing import Iterable, Tuple, List

# ===============================
# CONFIG (Denmark, UTM32N)
# ===============================
DK_X_MIN, DK_X_MAX = 350_000, 900_000
DK_Y_MIN, DK_Y_MAX = 6_060_000, 6_395_000
_WGS84_to_UTM32 = Transformer.from_crs(4326, 25832, always_xy=True)
YEAR = 365.25

# ===============================
# 1) DATE → SEASONAL COLUMNS
# ===============================
def add_seasonal_features(
    df: pd.DataFrame,
    date_col: str,
    ref_month: int = 2,
    ref_day: int = 1,
    harmonics: Tuple[int, ...] = (1, 2, 3),
    prefix: str = "season",
) -> pd.DataFrame:
    """Add cyclic seasonality features around a reference date (default Feb 1).
    Adds:
      - {prefix}_k{h}_sin / {prefix}_k{h}_cos  for each h in harmonics
      - {prefix}_abs  (abs circular distance scaled to [0,1])
      - {prefix}_missing  (1 if date missing)
    NaT/invalid → sin/cos=0, abs=0, missing=1
    """
    df = df.copy()
    dts = pd.to_datetime(df[date_col], errors="coerce")
    doy = dts.dt.dayofyear.astype("float32")  # NaN if NaT

    ref_doy = pd.Timestamp(year=2001, month=ref_month, day=ref_day).day_of_year
    sdist = (doy - ref_doy + YEAR / 2) % YEAR - YEAR / 2  # NaN stays NaN
    missing = sdist.isna().astype("float32")
    sdist_filled = sdist.fillna(0.0).astype("float32")

    for k in harmonics:
        theta = 2 * math.pi * k * (sdist_filled / YEAR)
        df[f"{prefix}_k{k}_sin"] = np.sin(theta).astype("float32")
        df[f"{prefix}_k{k}_cos"] = np.cos(theta).astype("float32")

    df[f"{prefix}_abs"] = (sdist_filled.abs() / (YEAR / 2)).clip(0, 1).astype("float32")
    df[f"{prefix}_missing"] = missing
    return df

def seasonal_feature_cols(harmonics: Tuple[int, ...] = (1, 2, 3), prefix="season") -> List[str]:
    cols = []
    for k in harmonics:
        cols += [f"{prefix}_k{k}_sin", f"{prefix}_k{k}_cos"]
    cols += [f"{prefix}_abs", f"{prefix}_missing"]
    return cols

# ===============================
# 2) LON/LAT → SPATIAL COLUMNS (DK, no-fit)
# ===============================
def _to_utm32_batch(lon: Iterable, lat: Iterable):
    lon = pd.to_numeric(pd.Series(lon), errors="coerce").astype(float).values
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").astype(float).values
    x, y = _WGS84_to_UTM32.transform(lon, lat)  # NaNs propagate
    return np.asarray(x, float), np.asarray(y, float)

def _norm_xy_batch(x: np.ndarray, y: np.ndarray):
    nx = (x - DK_X_MIN) / (DK_X_MAX - DK_X_MIN)
    ny = (y - DK_Y_MIN) / (DK_Y_MAX - DK_Y_MIN)
    return nx, ny

def _xy_fourier_batch(nx: np.ndarray, ny: np.ndarray, periods_km: Tuple[int, ...]):
    N = len(nx)
    out = np.zeros((N, 4 * len(periods_km)), dtype=np.float32)
    valid = ~np.isnan(nx) & ~np.isnan(ny)
    if not np.any(valid):
        return out
    Lx = DK_X_MAX - DK_X_MIN
    Ly = DK_Y_MAX - DK_Y_MIN
    idx = 0
    nx_v, ny_v = nx[valid], ny[valid]
    for P in periods_km:
        wx = 2 * math.pi * (Lx / (P * 1000.0)) * nx_v
        wy = 2 * math.pi * (Ly / (P * 1000.0)) * ny_v
        out[valid, idx + 0] = np.sin(wx)
        out[valid, idx + 1] = np.cos(wx)
        out[valid, idx + 2] = np.sin(wy)
        out[valid, idx + 3] = np.cos(wy)
        idx += 4
    return out

def _grid_ids_batch(x: np.ndarray, y: np.ndarray, cell_sizes_km: Tuple[int, ...]):
    N = len(x)
    gids = np.full((N, len(cell_sizes_km)), -1, dtype=np.int64)
    valid = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(valid):
        return gids
    xv, yv = x[valid], y[valid]
    for j, km in enumerate(cell_sizes_km):
        size = km * 1000.0
        gx = np.floor((xv - DK_X_MIN) / size).astype(np.int64)
        gy = np.floor((yv - DK_Y_MIN) / size).astype(np.int64)
        gids[valid, j] = gy * 1000 + gx  # compact 2D -> 1D
    return gids

def add_spatial_features(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    periods_km: Tuple[int, ...] = (400, 200, 100, 50, 25),
    grid_scales_km: Tuple[int, ...] = (10, 5, 2),
    prefix: str = "geo",
) -> pd.DataFrame:
    """Add spatial Fourier (multi-scale) + raw normalized coords + missing flags + grid IDs."""
    df = df.copy()
    x, y = _to_utm32_batch(df[lon_col], df[lat_col])
    nx, ny = _norm_xy_batch(x, y)

    # Fourier features
    F = _xy_fourier_batch(nx, ny, periods_km)
    fcols = []
    for P in periods_km:
        fcols += [
            f"{prefix}_P{P}km_sin_x",
            f"{prefix}_P{P}km_cos_x",
            f"{prefix}_P{P}km_sin_y",
            f"{prefix}_P{P}km_cos_y",
        ]
    df[fcols] = pd.DataFrame(F, columns=fcols, index=df.index)

    # Raw normalized + missing flags
    df[f"{prefix}_nx"] = np.where(np.isnan(nx), 0.0, nx).astype("float32")
    df[f"{prefix}_ny"] = np.where(np.isnan(ny), 0.0, ny).astype("float32")
    df[f"{prefix}_nx_missing"] = np.isnan(nx).astype("float32")
    df[f"{prefix}_ny_missing"] = np.isnan(ny).astype("float32")

    # Grid IDs
    G = _grid_ids_batch(x, y, grid_scales_km)
    for j, km in enumerate(grid_scales_km):
        df[f"{prefix}_grid{km}km_id"] = G[:, j]
    return df

def spatial_numeric_cols(periods_km: Tuple[int, ...] = (400, 200, 100, 50, 25), prefix="geo") -> List[str]:
    cols = []
    for P in periods_km:
        cols += [
            f"{prefix}_P{P}km_sin_x",
            f"{prefix}_P{P}km_cos_x",
            f"{prefix}_P{P}km_sin_y",
            f"{prefix}_P{P}km_cos_y",
        ]
    cols += [f"{prefix}_nx", f"{prefix}_ny", f"{prefix}_nx_missing", f"{prefix}_ny_missing"]
    return cols

def spatial_grid_cols(grid_scales_km: Tuple[int, ...] = (10, 5, 2), prefix="geo") -> List[str]:
    return [f"{prefix}_grid{km}km_id" for km in grid_scales_km]

# ===============================
# 3) EXAMPLE USAGE
# ===============================
if __name__ == "__main__":
    # Load
    meta_path = "../data/metadata/metadata.csv"
    df = pd.read_csv(meta_path)

    # Column names in your CSV (change if different)
    lon_col, lat_col, date_col = "Longitude", "Latitude", "eventDate"

    # Add features
    df = add_seasonal_features(df, date_col=date_col, harmonics=(1, 2, 3), prefix="season")
    df = add_spatial_features(
        df,
        lon_col=lon_col,
        lat_col=lat_col,
        periods_km=(400, 200, 100, 50, 25),
        grid_scales_km=(10, 5, 2),
        prefix="geo",
    )

    # Get column lists (handy for your Dataset later)
    SEASON_COLS = seasonal_feature_cols((1, 2, 3), prefix="season")
    GEO_NUM_COLS = spatial_numeric_cols((400, 200, 100, 50, 25), prefix="geo")
    GEO_GRID_COLS = spatial_grid_cols((10, 5, 2), prefix="geo")

    print("Added seasonal cols:", len(SEASON_COLS))
    print("Added spatial numeric cols:", len(GEO_NUM_COLS))
    print("Added spatial grid id cols:", len(GEO_GRID_COLS))

    # (Optional) Save a tidied version
    df.to_csv("../data/metadata/metadata_with_geo_time.csv", index=False)
