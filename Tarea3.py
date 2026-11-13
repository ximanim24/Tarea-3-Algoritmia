"""
Tarea 3 - Ajuste de la temperatura del Recocido Simulado (2-opt, TSP) con DOE
Opción A: factorial 3x3 + ANOVA de dos factores
Opción B: RSM (diseño central compuesto de 1 factor) + regresión cuadrática

Al correrlo te pregunta la opción; al terminar puedes elegir otra o salir.
Requiere: numpy, pandas, matplotlib, statsmodels
"""
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ============================================================
# CONFIGURACIÓN
# ============================================================
MATRICULA = 2225486
TAMANOS = [5, 10, 15]
# --- Parámetros que puedes editar para otro experimento -------------------
# Experimento 1 (el que ya corriste): T_MIN, T_MAX = 1.0, 1000.0 y COOLING_RATE = 0.99
# Experimento 2 (rango ajustado, donde T0 sí debería afectar el costo):
T_MIN, T_MAX = 1.0, 100.0    # rango de temperatura definido por mí
COOLING_RATE = 0.999         # enfriamiento más lento: T0 pesa más durante la búsqueda
MAX_ITER = 3000              # igual que en la Tarea 2
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 10000
TOL_VARIANZA_CERO = 1e-9     # si max-min del costo < tol, se considera sin variabilidad

# r se genera UNA sola vez, con la matrícula
random.seed(MATRICULA)
R = random.random()


# ============================================================
# RECOCIDO SIMULADO (tomado de la Tarea 2, Ejercicio 4)
# ============================================================
def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return cost


def get_neighbors_2opt(tour):
    n = len(tour)
    i = random.randint(1, n - 2)
    j = random.randint(i + 1, n - 1)
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def simulated_annealing(distance_matrix, cooling_schedule='exponential',
                        initial_temp=1000.0, cooling_rate=0.995, max_iter=2000):
    n = len(distance_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = calculate_tour_cost(current_tour, distance_matrix)

    best_tour = list(current_tour)
    best_cost = current_cost

    T = initial_temp
    history_costs = [current_cost]
    start_time = time.perf_counter()

    for _ in range(max_iter):
        new_tour = get_neighbors_2opt(current_tour)
        new_cost = calculate_tour_cost(new_tour, distance_matrix)
        delta = new_cost - current_cost

        # Criterio de aceptación de Metropolis
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-8)):
            current_tour = new_tour
            current_cost = new_cost
            if current_cost < best_cost:
                best_tour = list(current_tour)
                best_cost = current_cost

        history_costs.append(current_cost)

        if cooling_schedule == 'exponential':
            T *= cooling_rate
        elif cooling_schedule == 'linear':
            T -= (initial_temp / max_iter)
            if T < 0:
                T = 0

    elapsed_time = time.perf_counter() - start_time
    return best_tour, best_cost, history_costs, elapsed_time


def generar_instancia(n):
    """Una instancia fija por tamaño (la misma para todas las corridas)."""
    rng = np.random.default_rng(MATRICULA + n)
    coords = rng.random((n, 2)) * 100
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.linalg.norm(coords[i] - coords[j])
    return D


# ============================================================
# NIVELES DE TEMPERATURA SEGÚN LA OPCIÓN
# ============================================================
def definir_niveles(opcion):
    if opcion == "A":
        fracciones = {"bajo": 0.3 * R, "medio": 0.3 * R + 0.3, "alto": 0.3 * R + 0.6}
        replicas = {k: 10 for k in fracciones}
    else:
        fracciones = {
            "axial_bajo": 0.2 * R,
            "bajo": 0.2 * R + 0.2,
            "centro": 0.2 * R + 0.4,
            "alto": 0.2 * R + 0.6,
            "axial_alto": 0.2 * R + 0.8,
        }
        replicas = {k: (10 if k == "centro" else 5) for k in fracciones}
    niveles = {k: T_MIN + f * (T_MAX - T_MIN) for k, f in fracciones.items()}
    return niveles, replicas


def correr_experimento(niveles, replicas, salida):
    filas = []
    semilla = MATRICULA * 1000          # semilla distinta para cada corrida
    for n in TAMANOS:
        D = generar_instancia(n)
        for nivel, T0 in niveles.items():
            for rep in range(replicas[nivel]):
                semilla += 1
                random.seed(semilla)
                np.random.seed(semilla)
                _, costo, _, _ = simulated_annealing(
                    D, 'exponential', initial_temp=T0,
                    cooling_rate=COOLING_RATE, max_iter=MAX_ITER)
                filas.append({"tamano": n, "nivel": nivel, "temperatura": round(T0, 6),
                              "replica": rep + 1, "costo": costo})
    df = pd.DataFrame(filas)
    df.to_csv(f"{salida}/datos_crudos.csv", index=False)
    return df


def bootstrap_mediana(x, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.median(meds), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def resumen_bootstrap(df):
    filas = []
    for (n, T0, nivel), g in df.groupby(["tamano", "temperatura", "nivel"]):
        m, lo, hi = bootstrap_mediana(g["costo"])
        filas.append({"tamano": n, "nivel": nivel, "temperatura": T0,
                      "mediana_boot": m, "IC95_inf": lo, "IC95_sup": hi})
    return pd.DataFrame(filas)


# ============================================================
# OPCIÓN A
# ============================================================
def opcion_a(df, salida):
    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)
    print("\n=== ANOVA de dos factores (alfa = 0.05) ===\n", anova)
    anova.to_csv(f"{salida}/anova_dos_factores.csv")

    p_temp = anova.loc["C(temperatura)", "PR(>F)"]
    p_tam = anova.loc["C(tamano)", "PR(>F)"]
    p_int = anova.loc["C(temperatura):C(tamano)", "PR(>F)"]
    print(f"\np temperatura = {p_temp:.4g} | p tamaño = {p_tam:.4g} | p interacción = {p_int:.4g}")

    # Tamaños sin variabilidad (todas las corridas dan el mismo costo)
    degenerados = [n for n in TAMANOS
                   if np.ptp(df[df.tamano == n]["costo"]) < TOL_VARIANZA_CERO]
    if degenerados:
        print(f"\nAVISO: n = {degenerados} no tiene variabilidad (el algoritmo siempre llega "
              "al mismo costo). Esos tamaños no aportan información sobre la temperatura y "
              "rompen el supuesto de varianzas homogéneas del ANOVA.")
        restantes = [n for n in TAMANOS if n not in degenerados]
        if len(restantes) >= 2:
            df_r = df[df.tamano.isin(restantes)]
            anova_r = sm.stats.anova_lm(
                ols("costo ~ C(temperatura) * C(tamano)", data=df_r).fit(), typ=2)
            print(f"\n=== ANOVA complementario sin n = {degenerados} ===\n", anova_r)
            anova_r.to_csv(f"{salida}/anova_sin_tamanos_degenerados.csv")

    boot = resumen_bootstrap(df)
    boot["temp_signif"] = p_temp < 0.05
    boot["interaccion_signif"] = p_int < 0.05
    boot["sin_variabilidad"] = boot["tamano"].isin(degenerados)
    boot.to_csv(f"{salida}/bootstrap_por_punto.csv", index=False)
    print("\n=== Bootstrap por punto de diseño ===\n", boot.round(3).to_string(index=False))
    if not (p_temp < 0.05 or p_int < 0.05):
        print("\nNota: ni temperatura ni interacción fueron significativas; "
              "repórtalo así en el análisis.")

    # Efectos principales
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mt = df.groupby("temperatura")["costo"].mean()
    ax[0].plot(mt.index, mt.values, "o-")
    ax[0].set(title="Efecto principal: temperatura", xlabel="T0", ylabel="Costo medio")
    ms = df.groupby("tamano")["costo"].mean()
    ax[1].plot(ms.index, ms.values, "o-")
    ax[1].set(title="Efecto principal: tamaño", xlabel="n", ylabel="Costo medio")
    ax[0].grid(True, linestyle=":", alpha=0.6)
    ax[1].grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{salida}/efectos_principales.png", dpi=200)
    plt.close()

    # Interacción (un panel por tamaño con escala propia, porque el costo crece con n)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in TAMANOS:
        m = df[df.tamano == n].groupby("temperatura")["costo"].mean()
        ax[0].plot(m.index, m.values, "o-", label=f"n={n}")
        ax[1].plot(m.index, m.values / m.values.mean(), "o-", label=f"n={n}")
    ax[0].set(title="Interacción temperatura × tamaño", xlabel="T0", ylabel="Costo medio")
    ax[1].set(title="Interacción (costo relativo a la media de cada n)",
              xlabel="T0", ylabel="Costo / costo medio de su n")
    for a in ax:
        a.legend()
        a.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{salida}/interaccion.png", dpi=200)
    plt.close()

    mejores = boot.loc[boot.groupby("tamano")["mediana_boot"].idxmin()].copy()

    def conclusion(fila):
        if fila["sin_variabilidad"]:
            return "sin variabilidad: cualquier T0 da el mismo costo"
        if p_temp < 0.05 or p_int < 0.05:
            return "efecto de T0 significativo"
        return "T0 sin efecto significativo (menor mediana, no concluyente)"

    mejores["conclusion"] = mejores.apply(conclusion, axis=1)
    return mejores[["tamano", "temperatura", "mediana_boot", "IC95_inf", "IC95_sup",
                    "conclusion"]].rename(columns={"temperatura": "T_optima"})


# ============================================================
# OPCIÓN B
# ============================================================
def opcion_b(df, salida):
    filas = []
    boot = resumen_bootstrap(df)
    boot.to_csv(f"{salida}/bootstrap_por_punto.csv", index=False)
    fig, axes = plt.subplots(1, len(TAMANOS), figsize=(5.5 * len(TAMANOS), 4.5))
    for ax, n in zip(np.atleast_1d(axes), TAMANOS):
        d = df[df.tamano == n]
        if np.ptp(d["costo"]) < TOL_VARIANZA_CERO:
            print(f"\n=== n = {n} ===")
            print("AVISO: sin variabilidad (todas las corridas dan el mismo costo). "
                  "No se ajusta el modelo cuadrático: cualquier T0 es igual de bueno.")
            ax.scatter(d.temperatura, d.costo, s=15)
            ax.set(title=f"n = {n} (costo constante = {d.costo.iloc[0]:.3f})",
                   xlabel="T0", ylabel="Costo")
            ax.grid(True, linestyle=":", alpha=0.6)
            filas.append({"tamano": n, "T_optima": np.nan, "T_star_sin_recorte": np.nan,
                          "beta2": np.nan, "p_beta2": np.nan, "p_falta_ajuste": np.nan,
                          "nivel_evaluado_cercano": np.nan,
                          "mediana_boot": d.costo.iloc[0], "IC95_inf": d.costo.iloc[0],
                          "IC95_sup": d.costo.iloc[0],
                          "conclusion": "sin variabilidad: cualquier T0 da el mismo costo"})
            continue
        modelo = ols("costo ~ temperatura + I(temperatura**2)", data=d).fit()
        print(f"\n=== Regresión cuadrática, n = {n} ===")
        print(modelo.summary().tables[1])
        b0, b1, b2 = modelo.params
        p_b2 = modelo.pvalues["I(temperatura ** 2)"]

        modelo_completo = ols("costo ~ C(temperatura)", data=d).fit()
        p_lof = sm.stats.anova_lm(modelo, modelo_completo)["Pr(>F)"].iloc[1]
        print(f"β2 = {b2:.4g} | p(β2) = {p_b2:.4g} | p(falta de ajuste) = {p_lof:.4g}")

        t_lo, t_hi = d.temperatura.min(), d.temperatura.max()
        t_star = -b1 / (2 * b2) if b2 != 0 else np.nan
        if b2 > 0 and np.isfinite(t_star):
            t_opt = float(np.clip(t_star, t_lo, t_hi))   # dentro del rango evaluado
        else:
            print("  β2 <= 0: la parábola no tiene mínimo; se usa el nivel con menor mediana.")
            t_opt = d.groupby("temperatura")["costo"].median().idxmin()

        nivel_cercano = d.iloc[(d.temperatura - t_opt).abs().argmin()]["temperatura"]
        m, lo, hi = bootstrap_mediana(d[d.temperatura == nivel_cercano]["costo"])
        filas.append({"tamano": n, "T_optima": t_opt, "T_star_sin_recorte": t_star,
                      "beta2": b2, "p_beta2": p_b2, "p_falta_ajuste": p_lof,
                      "nivel_evaluado_cercano": nivel_cercano,
                      "mediana_boot": m, "IC95_inf": lo, "IC95_sup": hi,
                      "conclusion": ("curvatura significativa: T* confiable" if p_b2 < 0.05
                                     else "curvatura no significativa: T* no concluyente")})

        xs = np.linspace(t_lo, t_hi, 200)
        pred = modelo.get_prediction(pd.DataFrame({"temperatura": xs})).summary_frame(alpha=0.05)
        ax.scatter(d.temperatura, d.costo, s=12, alpha=0.5, label="observados")
        bn = boot[boot.tamano == n]
        ax.errorbar(bn.temperatura, bn.mediana_boot,
                    yerr=[bn.mediana_boot - bn.IC95_inf, bn.IC95_sup - bn.mediana_boot],
                    fmt="s", color="k", capsize=3, label="mediana boot + IC95")
        ax.plot(xs, pred["mean"], "r-", label="ajuste cuadrático")
        ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color="r", alpha=0.15)
        ax.axvline(t_opt, ls="--", color="g", label=f"T* = {t_opt:.1f}")
        ax.set(title=f"n = {n}", xlabel="T0", ylabel="Costo")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{salida}/curvas_ajustadas.png", dpi=200)
    plt.close()
    return pd.DataFrame(filas)


# ============================================================
# MENÚ
# ============================================================
def ejecutar(opcion):
    salida = f"resultados_opcion_{opcion}_T{T_MIN:g}-{T_MAX:g}_cr{COOLING_RATE:g}"
    os.makedirs(salida, exist_ok=True)
    niveles, replicas = definir_niveles(opcion)
    print(f"\nConfiguración: T en [{T_MIN:g}, {T_MAX:g}] | cooling_rate = {COOLING_RATE:g} "
          f"| max_iter = {MAX_ITER}")
    print(f"r = {R:.6f}")
    print("Niveles de temperatura:", {k: round(v, 3) for k, v in niveles.items()})
    print("Réplicas por nivel:", replicas)
    print("Corriendo experimento...")
    df = correr_experimento(niveles, replicas, salida)
    print(f"Corridas totales: {len(df)}")
    tabla = opcion_a(df, salida) if opcion == "A" else opcion_b(df, salida)
    tabla.to_csv(f"{salida}/tabla_resumen.csv", index=False)
    print("\n=== TABLA RESUMEN (para el reporte) ===")
    print(tabla.round(3).to_string(index=False))
    print(f"\nArchivos guardados en ./{salida}/")


if __name__ == "__main__":
    print("--- Tarea 3: Ajuste de temperatura del Recocido Simulado con DOE ---")
    while True:
        print("\nElige una opción:")
        print("  A) Diseño factorial completo (3 niveles, ANOVA de dos factores)")
        print("  B) Superficie de respuesta (5 niveles, regresión cuadrática)")
        print("  S) Salir")
        eleccion = input("Tu elección: ").strip().upper()
        if eleccion in ("A", "B"):
            ejecutar(eleccion)
        elif eleccion in ("S", "Q", ""):
            print("Fin del programa.")
            break
        else:
            print("Opción no válida, intenta de nuevo.")