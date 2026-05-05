# Analizador de Distribuciones con Fitter

Este script permite analizar un conjunto de datos y encontrar qué distribuciones estadísticas se ajustan mejor a ellos, proporcionando un ranking y una lista de las que fallaron.

## Instalación

Para instalar las dependencias necesarias, ejecuta:

```bash
pip install -r requirements.txt
```

## Uso

Puedes ejecutar el script directamente para ver un ejemplo con datos generados aleatoriamente:

```bash
python analyze_distributions.py
```

### Integración en tu propio código

Puedes importar la función `analyze_data` y pasarle tu lista de datos:

```python
from analyze_distributions import analyze_data

mis_datos = [1.2, 2.3, 1.8, 4.5, ...] # Tu lista de datos
analyze_data(mis_datos)
```

## Características

- **Ranking:** Muestra una tabla con las distribuciones ordenadas por el error cuadrático (Sum Square Error).
- **Distribuciones Fallidas:** Identifica automáticamente aquellas distribuciones que no pudieron converger o fallaron durante el ajuste.
- **Mejor Ajuste:** Indica cuál es la mejor distribución y sus parámetros óptimos (mu, sigma, etc.).
