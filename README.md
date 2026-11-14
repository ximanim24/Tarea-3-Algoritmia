[README.md](https://github.com/user-attachments/files/34832848/README.md)
# Tarea 3 – Ajuste de temperatura del Recocido Simulado con DOE

Algoritmia y Optimización – Gpo 032
Samantha Ximena De La Garza González (matrícula 2225486)

Ajuste de la temperatura inicial T0 del Recocido Simulado (2-opt) sobre TSP (n = 5, 10, 15)
con dos enfoques: **Opción A** (factorial completo + ANOVA de dos factores) y
**Opción B** (superficie de respuesta + regresión cuadrática). Se usa bootstrap para la
mediana del costo y su IC del 95 %.

## Cómo ejecutar

```bash
pip install -r requirements.txt
python tarea3_doe.py
```

El programa muestra un menú: `A`, `B` o `S` para salir. Cada opción guarda sus resultados
(gráficas PNG, CSV de datos crudos, ANOVA, bootstrap y tabla resumen) en su propia carpeta.

## Parámetros del experimento reportado

- Matrícula (semilla de r): 2225486 → r = 0.593410
- Rango de temperatura: `T_MIN, T_MAX = 1, 1000`
- Enfriamiento exponencial: `COOLING_RATE = 0.99`, `MAX_ITER = 3000`

Estos valores están al inicio de `tarea3_doe.py`. Las semillas son fijas, por lo que
los resultados son reproducibles.

## Contenido

- `tarea3_doe.py`: código completo (Recocido Simulado, diseño experimental, ANOVA, regresión, bootstrap y gráficas)
- Carpetas de resultados de la Opción A y la Opción B
- Reporte en PDF
