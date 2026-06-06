# Entrega 2 — Verificación de la distribución Fisk y corrida semanal

Scripts nuevos para **(1)** demostrar estadísticamente que la distribución
**Log-logística (Fisk)** se ajusta a la demanda real de empanadas de Caprese, y
**(2)** correr la simulación de demanda estructurada por semanas.

Todo separa la demanda en dos grupos, como el informe, cada uno con sus
parámetros de Stat::Fit:

| Grupo | c (forma) | loc | scale (β) |
| ----- | --------- | --- | --------- |
| Lunes a Viernes | 6,16 | -13,3 | 35,9 |
| Sábados y Domingos | 5,48 | -4,4 | 27,1 |

Los datos salen de `../capresse.csv` (columna `Uds. vendidas`), agrupados según
la columna `Dia`. Los scripts resuelven las rutas solos, así que corren desde
cualquier carpeta.

## Requisitos

Necesitan `numpy`, `scipy`, `pandas` y `matplotlib`. En este entorno **no hay
pip**; instalar con el gestor del sistema:

```bash
apt-get install -y python3-numpy python3-scipy python3-pandas python3-matplotlib
```

Ejecutar siempre con `python3` (no `python`).

## Scripts

### 1) `verificacion_fisk.py` — bondad de ajuste

Comprueba que la Fisk es adecuada con **cuatro** pruebas (dos pares: la versión
convencional + una versión que corrige el sesgo "optimista" de haber estimado
los parámetros con los mismos datos):

1. **K-S clásico** — Kolmogorov-Smirnov de una muestra (datos vs CDF teórica).
2. **K-S bootstrap** — p-valor por bootstrap paramétrico (re-ajusta la Fisk en
   cada réplica). Corrige el sesgo del K-S clásico.
3. **Chi² permisivo** — frecuencias observadas vs esperadas (Fisk discretizada
   con corrección de continuidad), `gl = celdas - 1`.
4. **Chi² estricto** — igual pero `gl = celdas - 1 - 3` (resta los 3 parámetros
   estimados).

Criterio del repo: **`p > 0,05` ⇒ NO se rechaza** (Fisk adecuada, 95%). Los
cuatro tests dan "no rechazada" en ambos grupos.

```bash
python3 verificacion_fisk.py                 # capresse.csv del repo, bootstrap B=300
python3 verificacion_fisk.py --bootstrap 0   # sin bootstrap (instantáneo)
python3 verificacion_fisk.py otros_datos.csv
```

> El bootstrap re-ajusta una Fisk por réplica, por eso tarda ~40 s con B=300.
> Usá `--bootstrap 0` si querés solo el K-S clásico + Chi².

### 2) `grafico_cdf_fisk.py` — CDF empírica vs teórica

Dibuja, por grupo, la **CDF empírica** de los datos (escalera) contra la **CDF
teórica de Fisk** (curva), y marca en verde la distancia **D** del K-S (la
máxima separación vertical). Si las curvas van pegadas, la Fisk ajusta bien.

Usa backend `Agg` (no abre ventana) y guarda **`cdf_fisk.png`** junto al script.

```bash
python3 grafico_cdf_fisk.py        # genera cdf_fisk.png
```

### 3) `corrida_fisk_semanal.py` — corrida (historia ficticia)

Genera demanda simulada por **transformada inversa** Fisk, estructurada por
**semanas** de 7 días (5 de Lunes a Viernes + 2 de Sábado y Domingo), cada día
con los parámetros de su grupo. Reporta la demanda por día y por semana, y
guarda el detalle en **`simulacion_fisk_semanal.csv`**.

La fuente de los números pseudoaleatorios U(0,1) es configurable:

```bash
python3 corrida_fisk_semanal.py                 # 4 semanas con el GLC propio
python3 corrida_fisk_semanal.py 8               # 8 semanas con el GLC
python3 corrida_fisk_semanal.py --semilla 123 8 # GLC con otra semilla
python3 corrida_fisk_semanal.py ../numeros.csv  # usa los U del archivo (len//7 semanas)
```

Este script **solo genera**; la validación de la distribución está en
`verificacion_fisk.py`.

## Salidas generadas (no versionar)

- `cdf_fisk.png`
- `simulacion_fisk_semanal.csv`

## Nota sobre los parámetros

Estos scripts usan los **dos** juegos de parámetros del informe (uno por tipo de
día). La corrida vieja de la raíz (`../corrida_fisk.py`) usaba un único juego
(`c=6.22, loc=-12, scale=34.7`) sobre toda la serie, que no figura en el informe.
