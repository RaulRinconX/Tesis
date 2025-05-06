#!/usr/bin/env python3
"""
detectar_dron.py
Detecta la presencia de un dron a partir de mediciones RF en CSV.

Uso:
    python detectar_dron.py baseline.csv data.csv [--stats]

Posicionales
------------
baseline.csv : datos con el dron APAGADO (calibración)
data.csv     : datos a evaluar (¿dron presente?)

Opciones
--------
--stats      : imprime estadísticas y umbrales empleados
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

# ---------------------- PARSING DE ARGUMENTOS ------------------------------- #
parser = argparse.ArgumentParser(description="Detección de dron basada en Power Flux Density y potencia total.")
parser.add_argument("baseline",  help="CSV con el dron apagado (calibra umbrales).")
parser.add_argument("datafile",  help="CSV a analizar (detección).")
parser.add_argument("-s", "--stats", action="store_true", help="Muestra estadísticas de ambos archivos.")
args = parser.parse_args()

# ------------------------ FUNCIONES AUXILIARES ----------------------------- #
def leer_csv(ruta: str) -> pd.DataFrame:
    """Lee un CSV intentando UTF‑8 y, si falla, Latin‑1."""
    ruta = Path(ruta)
    if not ruta.exists():
        print(f"ERROR: no se encontró el archivo '{ruta}'.")
        sys.exit(1)
    try:
        return pd.read_csv(ruta)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="ISO-8859-1")

def buscar_columna(df: pd.DataFrame, clave: str) -> str | None:
    """Devuelve la primera columna que contenga la clave dada (case‑insensitive)."""
    for col in df.columns:
        if clave.lower() in col.lower():
            return col
    return None

# ------------------------ CARGA DE DATOS ------------------------------------ #
df_base = leer_csv(args.baseline)
df_data = leer_csv(args.datafile)

pfd_col_base  = buscar_columna(df_base,  "Power Flux Density")
pfd_col_data  = buscar_columna(df_data,  "Power Flux Density")
pow_col_base  = buscar_columna(df_base,  "Total Spectrum Power")
pow_col_data  = buscar_columna(df_data,  "Total Spectrum Power")

if not pfd_col_base or not pfd_col_data:
    print("ERROR: No se encontro la columna 'Power Flux Density' en uno de los archivos.")
    sys.exit(1)

# --------------------- CALCULO DE UMBRALES ---------------------------------- #
mean_pfd_base = df_base[pfd_col_base].mean()
std_pfd_base  = df_base[pfd_col_base].std()
umbral_pfd    = mean_pfd_base + 3 * std_pfd_base

umbral_power = None
if pow_col_base and pow_col_data:
    mean_pow_base = df_base[pow_col_base].mean()
    std_pow_base  = df_base[pow_col_base].std()
    umbral_power  = mean_pow_base + 3 * std_pow_base

# -------------------------- DETECCION --------------------------------------- #
if umbral_power is not None:
    mask = (df_data[pfd_col_data] > umbral_pfd) & (df_data[pow_col_data] > umbral_power)
else:
    mask = df_data[pfd_col_data] > umbral_pfd

dron_detectado = mask.any()

# ------------------------ RESULTADOS / STATS -------------------------------- #
if args.stats:
    print("\n--- ESTADISTICAS (baseline) ---")
    print(f"{pfd_col_base}: μ={mean_pfd_base:.2f}, σ={std_pfd_base:.2f} ⇒ umbral={umbral_pfd:.2f}")
    if umbral_power is not None:
        print(f"{pow_col_base}: μ={mean_pow_base:.2f} dBm, σ={std_pow_base:.2f} ⇒ umbral={umbral_power:.2f} dBm")
    print("\n--- ESTADISTICAS (data) ---")
    print(df_data[[c for c in [pfd_col_data, pow_col_data] if c]].describe().T)

print("\n===== RESULTADO =====")
print("Dron detectado" if dron_detectado else "Sin dron")
