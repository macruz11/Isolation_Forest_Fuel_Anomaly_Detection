
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de consumo de combustible listo para ejecutarse como script en VS Code.

Principales cambios vs. versión Kaggle/Jupyter:
- Elimina dependencias del entorno de Kaggle y rutas de iOS/Juno.
- Parametriza rutas y fuentes de datos con argparse.
- Evita credenciales hardcoded: usa variables de entorno o un DSN ODBC.
- Estructura en funciones + main(), logs claros y manejo de errores.

Requisitos (instalar en tu entorno):
    pandas, numpy, scikit-learn, openpyxl, plotly
    (opcional) pyodbc si deseas leer desde ODBC/SAP Datasphere.
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas import ExcelWriter

# Modelado / features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Visualización interactiva
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------
# Utilidades generales
# ---------------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------
# Selección de archivo (GUI opcional)
# ---------------------------------
def pick_csv_file_dialog(title: str = "Selecciona el archivo CSV de entrada") -> Path | None:
    """Abre un selector de archivos en Windows (y otros SO con Tk).

    - Se fuerza la ventana a quedar por encima (topmost) para evitar que quede detrás.
    - Retorna Path del archivo o None si el usuario cancela.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    # Asegurar topmost
    try:
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        root.update()
    except Exception:
        pass

    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    )
    try:
        root.destroy()
    except Exception:
        pass

    if not file_path:
        return None
    return Path(file_path)

# ---------------------------------
# Carga de datos
# ---------------------------------

def read_csv_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo CSV: {path}")
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        raise ValueError("El CSV debe contener una columna 'date'.")
    df['date'] = pd.to_datetime(df['date'])
    return df


def read_from_odbc(dsn: str, uid: str | None, pwd: str | None, schema: str, view_name: str, limit: int | None = None) -> pd.DataFrame:
    try:
        import pyodbc  # opcional
    except Exception as e:
        raise RuntimeError("pyodbc no está disponible. Instálalo para usar ODBC.") from e

    parts = [f"DSN={dsn};"]
    if uid:
        parts.append(f"UID={uid};")
    if pwd:
        parts.append(f"PWD={pwd}")
    conn_str = ''.join(parts)

    log(f"Conectando a ODBC DSN='{dsn}' schema='{schema}' view='{view_name}'...")
    with pyodbc.connect(conn_str) as conn:
        # Validar existencia de la vista (ajusta la consulta según tu motor)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT VIEW_NAME
                FROM PUBLIC.VIEWS
                WHERE SCHEMA_NAME = ? AND VIEW_NAME = ?
                """,
                (schema, view_name),
            )
            exists = cursor.fetchone()
            if not exists:
                raise ValueError(f"La vista {schema}.{view_name} no existe o no hay permisos.")
        except Exception:
            # Si tu motor no tiene PUBLIC.VIEWS, intenta leer directo.
            pass

        q = f'SELECT * FROM "{schema}"."{view_name}"'
        if limit is not None and limit > 0:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql(q, conn)

    if 'date' not in df.columns:
        raise ValueError("La vista debe contener una columna 'date'.")
    df['date'] = pd.to_datetime(df['date'])
    return df

# ---------------------------------
# Pipeline: inyección/etiquetado, features, modelo
# ---------------------------------
ANOMALY_TYPES = ['point_spike','point_drop','level_shift','variance_burst','contextual_ratio']


def inject_and_label(
    df: pd.DataFrame,
    equipment_col: str = 'equipment_id',
    date_col: str = 'date',
    value_col: str = 'fuel_gallons_per_hour',
    random_state: int = 42,
    rate_point: float = 0.006,
    rate_levelshift: float = 0.01,
    rate_varburst: float = 0.005,
    rate_contextual: float = 0.004,
    spike_multiplier_range: Tuple[float, float] = (3.0, 6.0),
    drop_multiplier_range: Tuple[float, float] = (0.1, 0.4),
    levelshift_multiplier_range: Tuple[float, float] = (1.25, 1.6),
    varburst_multiplier_range: Tuple[float, float] = (1.8, 3.0),
    context_ratio_multiplier_range: Tuple[float, float] = (2.0, 4.0),
    min_series_len: int = 60,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values([equipment_col, date_col], inplace=True)

    # columnas de etiquetas
    df['label'] = 0
    df['anomaly_type'] = pd.Series(index=df.index, dtype='object')
    df['label_source'] = pd.Series(index=df.index, dtype='object')
    df['orig_value'] = df[value_col].astype(float)

    has_hours = 'hours_operated' in df.columns
    has_distance = 'distance_km' in df.columns

    out_chunks: List[pd.DataFrame] = []

    for eq, g in df.groupby(equipment_col):
        g = g.sort_values(date_col).copy()
        n = len(g)

        def set_label(idx, type_name: str):
            idx = pd.Index(idx)
            mask = g.loc[idx, 'label'].eq(0)
            idx_final = idx[mask]
            g.loc[idx_final, 'label'] = 1
            g.loc[idx_final, 'anomaly_type'] = type_name
            g.loc[idx_final, 'label_source'] = 'synthetic'

        # point anomalies
        n_point = int(np.ceil(rate_point * n))
        if n_point > 0:
            idxs = rng.choice(g.index, size=n_point, replace=False)
            half = n_point // 2
            spike_idxs = idxs[:half]
            drop_idxs = idxs[half:]

            spikes = rng.uniform(*spike_multiplier_range, size=len(spike_idxs))
            g.loc[spike_idxs, value_col] *= spikes
            set_label(spike_idxs, 'point_spike')

            drops = rng.uniform(*drop_multiplier_range, size=len(drop_idxs))
            g.loc[drop_idxs, value_col] *= drops
            set_label(drop_idxs, 'point_drop')

        # level shift
        if n >= min_series_len and rate_levelshift > 0:
            seg_len = rng.integers(low=7, high=max(8, min(28, n//4)))
            n_seg = max(1, int(rate_levelshift * n / seg_len))
            starts = rng.choice(range(0, n - seg_len), size=n_seg, replace=False)
            for s in starts:
                idx_seg = g.index[s:s+seg_len]
                mult = rng.uniform(*levelshift_multiplier_range)
                g.loc[idx_seg, value_col] *= mult
                set_label(idx_seg, 'level_shift')

        # variance burst
        if n >= min_series_len and rate_varburst > 0:
            win_len = rng.integers(low=7, high=max(8, min(21, n//5)))
            n_win = max(1, int(rate_varburst * n / win_len))
            starts = rng.choice(range(0, n - win_len), size=n_win, replace=False)
            for s in starts:
                idx_win = g.index[s:s+win_len]
                local_mean = g.loc[idx_win, value_col].mean()
                mult = rng.uniform(*varburst_multiplier_range)
                g.loc[idx_win, value_col] = local_mean + (g.loc[idx_win, value_col] - local_mean) * mult
                set_label(idx_win, 'variance_burst')

        # contextual ratio
        if (has_hours or has_distance) and rate_contextual > 0:
            n_ctx = int(np.ceil(rate_contextual * n))
            if n_ctx > 0:
                ctx_idxs = rng.choice(g.index, size=n_ctx, replace=False)
                mult = rng.uniform(*context_ratio_multiplier_range, size=n_ctx)
                g.loc[ctx_idxs, value_col] *= mult
                set_label(ctx_idxs, 'contextual_ratio')

        # sane values
        g[value_col] = g[value_col].clip(lower=0)
        out_chunks.append(g)

    return pd.concat(out_chunks).sort_values([equipment_col, date_col]).reset_index(drop=True)


def apply_business_rule_labels(
    df: pd.DataFrame,
    capacity_map: Dict[str, float] | None = None,
    equipment_col: str = 'equipment_id',
    date_col: str = 'date',
    value_col: str = 'fuel_gallons_per_hour',
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if 'label' not in df.columns:
        df['label'] = 0
    if 'anomaly_type' not in df.columns:
        df['anomaly_type'] = 'none'
    if 'label_source' not in df.columns:
        df['label_source'] = 'none'

    if capacity_map is not None:
        df['tank_capacity'] = df[equipment_col].map(capacity_map)
        overcap = (df['tank_capacity'].notna()) & (df[value_col] > df['tank_capacity'])
        df.loc[overcap, 'label'] = 1
        df.loc[overcap, 'anomaly_type'] = 'rule_capacity'
        df.loc[overcap, 'label_source'] = 'rule'
    return df


def cyclic_encode(series: pd.Series, period: int) -> np.ndarray:
    angle = 2 * np.pi * (series % period) / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def engineer_features(
    df: pd.DataFrame,
    equipment_col: str = 'equipment_id',
    date_col: str = 'date',
    value_col: str = 'fuel_gallons_per_hour',
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values([equipment_col, date_col], inplace=True)

    out: List[pd.DataFrame] = []
    for eq, g in df.groupby(equipment_col):
        g = g.sort_values(date_col).copy()
        g['lag_1'] = g[value_col].shift(1)
        g['lag_7'] = g[value_col].shift(7)
        g['ma_7'] = g[value_col].rolling(7, min_periods=3).mean()
        g['ma_14'] = g[value_col].rolling(14, min_periods=5).mean()
        g['rolling_std_7'] = g[value_col].rolling(7, min_periods=3).std()
        g['ratio_to_ma7'] = g[value_col] / (g['ma_7'] + 1e-6)
        g['dow'] = g[date_col].dt.dayofweek
        g['month'] = g[date_col].dt.month
        dow_enc = cyclic_encode(g['dow'].values, period=7)
        mon_enc = cyclic_encode((g['month']-1).values, period=12)
        g[['dow_sin','dow_cos']] = pd.DataFrame(dow_enc, index=g.index)
        g[['mon_sin','mon_cos']] = pd.DataFrame(mon_enc, index=g.index)
        out.append(g)

    feats = pd.concat(out, axis=0)
    feats = feats.dropna(subset=['lag_1','ma_7']).copy()
    feature_cols = [
        value_col,'lag_1','lag_7','ma_7','ma_14','rolling_std_7','ratio_to_ma7',
        'dow_sin','dow_cos','mon_sin','mon_cos'
    ]
    return feats, feature_cols



def align_labels_after_features(
    feats: pd.DataFrame,
    original_with_labels: pd.DataFrame,
    equipment_col: str = 'equipment_id',
    date_col: str = 'date',
    value_col: str = 'fuel_gallons_per_hour',
) -> pd.DataFrame:
    """Alinea etiquetas con el dataframe de features evitando duplicados.

    Problema típico:
      - feats puede contener ya columnas de etiquetas (label/anomaly_type/label_source)
      - el merge genera sufijos (_x/_y) y luego se rompe al acceder a 'label'.

    Solución:
      - eliminar columnas de etiquetas de feats antes del merge
      - hacer merge solo de etiquetas (y del value_col solo si no existe en feats)
    """
    feats = feats.copy()
    feats[date_col] = pd.to_datetime(feats[date_col])

    # Eliminar columnas que causarían sufijos o ambigüedad
    drop_cols = ['label', 'anomaly_type', 'label_source', 'orig_value']
    feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns], errors='ignore')

    original = original_with_labels.copy()
    original[date_col] = pd.to_datetime(original[date_col])

    cols_to_merge = [equipment_col, date_col, 'label', 'anomaly_type', 'label_source']
    # Solo traer value_col si no está ya en feats
    if value_col not in feats.columns and value_col in original.columns:
        cols_to_merge.append(value_col)

    merged = feats.merge(
        original[cols_to_merge],
        on=[equipment_col, date_col],
        how='left'
    )

    merged['label'] = merged['label'].fillna(0).astype(int)
    merged['anomaly_type'] = merged['anomaly_type'].fillna('none')
    merged['label_source'] = merged['label_source'].fillna('none')
    return merged


def detect_anomalies_isoforest(
    df: pd.DataFrame,
    feature_cols: List[str],
    equipment_col: str = 'equipment_id',
    date_col: str = 'date',
    value_col: str = 'fuel_gallons_per_hour',
    contamination: float = 0.01,
) -> pd.DataFrame:
    results: List[pd.DataFrame] = []
    for eq, g in df.groupby(equipment_col):
        X = g[feature_cols].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(
            n_estimators=400,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        iso.fit(Xs)
        decision = iso.decision_function(Xs)
        score = -decision
        pred = iso.predict(Xs)

        cols = [equipment_col, date_col, value_col] + feature_cols
        # evitar duplicados en caso de overlapping
        seen, cols_unique = set(), []
        for c in cols:
            if c not in seen:
                seen.add(c)
                cols_unique.append(c)
        g_out = g[cols_unique].copy()
        g_out['anomaly_flag'] = (pred == -1).astype(int)
        g_out['anomaly_score'] = score
        g_out['model'] = 'IsolationForest'
        results.append(g_out)
    return pd.concat(results, axis=0).sort_values([equipment_col, date_col])

# ---------------------------------
# Gráficos y exportación
# ---------------------------------

def build_iso_dashboard(iso_scored: pd.DataFrame, value_col: str = 'fuel_gallons_per_hour') -> str:
    fig_iso = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=("Consumo (galones EE.UU./hora) con anomalías", "Score Isolation Forest")
    )
    fig_iso.add_trace(
        go.Scatter(x=iso_scored['date'], y=iso_scored[value_col],
                   mode='lines+markers', name='Fuel', line=dict(color='gray')),
        row=1, col=1,
    )
    anomalies = iso_scored[iso_scored['anomaly_flag'] == 1]
    fig_iso.add_trace(
        go.Scatter(x=anomalies['date'], y=anomalies[value_col],
                   mode='markers', name='Anomalías', marker=dict(color='red', size=8)),
        row=1, col=1,
    )
    fig_iso.add_trace(
        go.Scatter(x=iso_scored['date'], y=iso_scored['anomaly_score'],
                   mode='lines', name='Score', line=dict(color='blue')),
        row=2, col=1,
    )
    fig_iso.update_layout(title_text="Isolation Forest", height=600)
    # Devuelve HTML del gráfico (sin plotly.js para incrustar en plantilla)
    return fig_iso.to_html(full_html=False, include_plotlyjs=False)


def write_html_dashboard(out_dir: Path, iso_html: str) -> Path:
    html_template = f"""
<!DOCTYPE html>
<html lang='es'>
<head>
  <meta charset='utf-8'>
  <title>Fuel Anomaly Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: Arial; margin: 0; padding: 0; }}
    header {{ background: #f7f7f7; padding: 10px; text-align: center; }}
    .panel {{ margin: 20px; }}
  </style>
</head>
<body>
  <header><h2>Fuel Anomaly Dashboard</h2></header>
  <div class='panel'>{iso_html}</div>
</body>
</html>
"""
    html_path = out_dir / "interactive_dashboard.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    return html_path


def export_results(out_dir: Path, df_labeled: pd.DataFrame, iso_scored: pd.DataFrame) -> None:
    out_dir = ensure_dir(out_dir)
    iso_csv = out_dir / "fuel_anomalies_isoforest.csv"
    iso_scored.to_csv(iso_csv, index=False)
    log(f"CSV de resultados: {iso_csv}")

    excel_path = out_dir / "Fuel_Anomaly_Results.xlsx"
    with ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_labeled.to_excel(writer, sheet_name='Labeled_Data', index=False)
        iso_scored.to_excel(writer, sheet_name='IsoForest_Scored', index=False)
    log(f"Excel guardado en: {excel_path}")

# ---------------------------------
# main()
# ---------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline de anomalías de consumo de combustible",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--input-csv", type=Path, help="Ruta a CSV de entrada")
    src.add_argument("--odbc-dsn", type=str, help="Nombre del DSN ODBC para leer desde una vista")

    p.add_argument("--odbc-uid", type=str, default=os.getenv("ODBC_UID"), help="Usuario ODBC (si aplica)")
    p.add_argument("--odbc-pwd", type=str, default=os.getenv("ODBC_PWD"), help="Password ODBC (si aplica)")
    p.add_argument("--schema", type=str, default=os.getenv("ODBC_SCHEMA", "DESARROLLO_TD"), help="Schema de la vista (ODBC)")
    p.add_argument("--view", type=str, default=os.getenv("ODBC_VIEW", "Vista_materiales_completa"), help="Nombre de la vista (ODBC)")
    p.add_argument("--limit", type=int, default=None, help="LIMIT para consulta a la vista")

    p.add_argument("--out-dir", type=Path, default=Path.cwd() / "FuelModelResults", help="Carpeta de salida")

    p.add_argument("--seed", type=int, default=2025, help="Semilla para inyección de anomalías")
    p.add_argument("--contamination", type=float, default=0.01, help="Contaminación estimada para IsolationForest")

    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    # 0) Si no se pasa --input-csv ni --odbc-dsn, abrir selector de archivo
    if (not args.input_csv) and (not getattr(args, 'odbc_dsn', None)):
        picked = pick_csv_file_dialog()
        if picked:
            args.input_csv = picked
            log(f"Archivo seleccionado por GUI: {args.input_csv}")
        else:
            raise ValueError("No se seleccionó ningún CSV y no se proporcionó fuente de datos.")

    # 1) Cargar datos
    if args.input_csv:
        log(f"Cargando datos desde CSV: {args.input_csv}")
        df = read_csv_input(args.input_csv)
    else:
        # ODBC
        uid = args.odbc_uid
        pwd = args.odbc_pwd
        if not uid or not pwd:
            warn("No se recibieron UID/PWD. Intentaré sólo con el DSN, si el DSN tiene autenticación integrada.")
        df = read_from_odbc(args.odbc_dsn, uid, pwd, args.schema, args.view, args.limit)
        log(f"Datos leídos desde ODBC DSN={args.odbc_dsn} schema={args.schema} view={args.view}")

    # Validaciones mínimas
    must_cols = {'equipment_id', 'fuel_gallons_per_hour', 'date'}
    missing = must_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    # 2) Inyectar y etiquetar
    log("Inyectando anomalías...")
    df_syn = inject_and_label(df, random_state=args.seed)

    # 3) Reglas de negocio (opcional: agrega un mapa si lo tienes)
    log("Aplicando reglas de negocio...")
    df_labeled = apply_business_rule_labels(df_syn)

    # 4) Ingeniería de características
    log("Ingeniería de características...")
    feats, feature_cols = engineer_features(df_labeled)
    feats_labeled = align_labels_after_features(feats, df_labeled)

    # 5) Modelo Isolation Forest
    log("Entrenando y puntuando con Isolation Forest...")
    iso_scored = detect_anomalies_isoforest(feats_labeled, feature_cols, contamination=args.contamination)

    # 6) Exportar resultados
    out_dir = ensure_dir(args.out_dir)
    export_results(out_dir, df_labeled, iso_scored)

    # 7) Dashboard HTML
    log("Generando dashboard interactivo...")
    iso_html = build_iso_dashboard(iso_scored)
    html_path = write_html_dashboard(out_dir, iso_html)
    log(f"Dashboard interactivo: {html_path}")

    log("Listo ✅")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        err(str(e))
        raise
