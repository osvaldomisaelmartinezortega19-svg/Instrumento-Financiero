
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, kurtosis

np.random.seed(42)

# Función para limpiar números
def limpiar_numero(texto):
    try:
        texto = str(texto).strip()
        if "," in texto and "." in texto:
            texto = texto.replace(",", "")
        elif "," in texto and "." not in texto:
            if texto.count(",") == 1 and len(texto.split(",")[-1]) <= 2:
                texto = texto.replace(",", ".")
            else:
                texto = texto.replace(",", "")
        return float(texto)
    except:
        return 0.0

# === Parámetros globales ===
num_activos = int(limpiar_numero(input("¿Cuántos activos quieres simular? ") or "1"))
num_simulaciones = int(limpiar_numero(input("¿Cuántas simulaciones por activo? ") or "1000"))
inflacion = limpiar_numero(input("Inflación anual estimada (%) ") or "3")/100
tasa_impuestos = limpiar_numero(input("Tasa de impuestos anual sobre rendimientos (%) ") or "10")/100

activos = []
for i in range(num_activos):
    print(f"\n--- Activo {i+1} ---")
    nombre = input("Nombre del activo: ") or f"Activo {i+1}"
    capital = limpiar_numero(input("Capital inicial ($): ") or "10000")
    rendimiento_esperado = limpiar_numero(input("Rendimiento anual esperado (%): ") or "8")/100
    años = int(limpiar_numero(input("Tiempo (años): ") or "10"))
    contribucion = limpiar_numero(input("Contribución anual ($, 0 si no aplica): ") or "0")
    desviacion = rendimiento_esperado * 0.2

    activos.append({
        "nombre": nombre,
        "capital": capital,
        "rendimiento_esperado": rendimiento_esperado,
        "años": años,
        "contribucion": contribucion,
        "desviacion": desviacion
    })

# === Simulación Monte Carlo ===
plt.close('all')
fig, axs = plt.subplots(len(activos), 4, figsize=(24, 5*len(activos)))
if len(activos) == 1:
    axs = np.expand_dims(axs, axis=0)

resultados = []

for idx, activo in enumerate(activos):
    simulaciones = []
    contribuciones_acum = []

    for s in range(num_simulaciones):
        valores = [activo["capital"]]
        contrib_acum = [0]
        max_valor = activo["capital"]

        for t in range(1, activo["años"]+1):
            r = np.random.normal(activo["rendimiento_esperado"], activo["desviacion"])
            ganancia = valores[-1]*r
            impuestos = ganancia * tasa_impuestos
            nuevo_valor = valores[-1] + ganancia - impuestos + activo["contribucion"]
            valores.append(nuevo_valor)
            contrib_acum.append(contrib_acum[-1] + activo["contribucion"])
            max_valor = max(max_valor, nuevo_valor)

        simulaciones.append(valores[1:])
        contribuciones_acum.append(contrib_acum[1:])

    simulaciones = np.array(simulaciones)
    contribuciones_acum = np.array(contribuciones_acum)
    finales_reales = simulaciones[:,-1]/((1+inflacion)**activo["años"])

    # Métricas estadísticas
    var_5 = np.percentile(finales_reales,5)
    var_1 = np.percentile(finales_reales,1)
    cvar_5 = finales_reales[finales_reales <= var_5].mean()
    cvar_1 = finales_reales[finales_reales <= var_1].mean()
    volatilidad = np.std(finales_reales)
    varianza = np.var(finales_reales)
    skewness = skew(finales_reales)
    kurt_val = kurtosis(finales_reales)
    contrib_total = contribuciones_acum[:,-1].mean()
    max_drawdown = np.max([
        np.max(simulaciones[s] - np.maximum.accumulate(simulaciones[s]))
        for s in range(num_simulaciones)
    ])
    sharpe = (np.mean(finales_reales)/volatilidad) if volatilidad>0 else 0

    resultados.append({
        "Activo": activo["nombre"],
        "Media ($)": finales_reales.mean(),
        "Mediana ($)": np.median(finales_reales),
        "Min ($)": finales_reales.min(),
        "Max ($)": finales_reales.max(),
        "Desv.Std ($)": volatilidad,
        "Varianza ($)": varianza,
        "Skewness": skewness,
        "Kurtosis": kurt_val,
        "VaR 5% ($)": var_5,
        "CVaR 5% ($)": cvar_5,
        "VaR 1% ($)": var_1,
        "CVaR 1% ($)": cvar_1,
        "Drawdown Máx ($)": max_drawdown,
        "Sharpe ratio": sharpe,
        "Contribución Total ($)": contrib_total
    })

    # === Gráficas ===
    promedio = simulaciones.mean(axis=0)
    p5 = np.percentile(simulaciones,5,axis=0)
    p95 = np.percentile(simulaciones,95,axis=0)
    valores_reales_prom = promedio / ((1+inflacion)**np.arange(1, activo["años"]+1))

    axs[idx,0].plot(range(1, activo["años"]+1), valores_reales_prom, marker='o', label="Promedio (real)")
    axs[idx,0].fill_between(range(1, activo["años"]+1),
                            p5/((1+inflacion)**np.arange(1, activo["años"]+1)),
                            p95/((1+inflacion)**np.arange(1, activo["años"]+1)),
                            alpha=0.2, color="orange", label="5%-95%")
    axs[idx,0].set_title(f"{activo['nombre']}: Valor real promedio y rango")
    axs[idx,0].set_xlabel("Años")
    axs[idx,0].set_ylabel("Valor ($)")
    axs[idx,0].legend()
    axs[idx,0].grid(True)

    axs[idx,1].hist(finales_reales, bins=25, alpha=0.7, density=True, color="skyblue")
    axs[idx,1].set_title(f"{activo['nombre']}: Distribución valor final real")
    axs[idx,1].set_xlabel("Valor final ($)")
    axs[idx,1].set_ylabel("Densidad")
    axs[idx,1].grid(True)

    axs[idx,2].boxplot(finales_reales, vert=True, patch_artist=True, labels=[activo["nombre"]])
    axs[idx,2].set_title(f"{activo['nombre']}: Boxplot escenarios finales")
    axs[idx,2].grid(True)

    sorted_vals = np.sort(finales_reales)
    ecdf = np.arange(1, len(sorted_vals)+1)/len(sorted_vals)
    axs[idx,3].plot(sorted_vals, ecdf, marker='.', linestyle='none')
    axs[idx,3].set_title(f"{activo['nombre']}: Distribución acumulada (ECDF)")
    axs[idx,3].set_xlabel("Valor final ($)")
    axs[idx,3].set_ylabel("Probabilidad acumulada")
    axs[idx,3].grid(True)

plt.tight_layout()
plt.show()

# === Informe ===
df_resultados = pd.DataFrame(resultados)
print("\n📊 Informe estadístico completo por activo (ajustado a inflación):\n")
pd.set_option('display.float_format', '${:,.2f}'.format)
print(df_resultados.to_string(index=False))

print(f"\n💰 Valor esperado del portafolio (suma de medias ajustadas): ${df_resultados['Media ($)'].sum():,.2f}")
