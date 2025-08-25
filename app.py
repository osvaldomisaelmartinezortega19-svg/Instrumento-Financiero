import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
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
        return 0

st.set_page_config(page_title="Simulador Financiero", page_icon="💰", layout="wide")
st.title("Simulador Financiero Monte Carlo con Análisis de Riesgo")

# Parámetros generales
num_activos = st.number_input("Número de activos", min_value=1, value=1)
num_simulaciones = st.number_input("Número de simulaciones por activo", min_value=100, value=1000)
inflacion = st.number_input("Inflación anual estimada (%)", value=3.0)/100
tasa_impuestos = st.number_input("Tasa de impuestos anual sobre rendimientos (%)", value=10.0)/100

activos = []
st.subheader("Definir cada activo")
for i in range(num_activos):
    st.markdown(f"### Activo {i+1}")
    nombre = st.text_input(f"Nombre del activo {i+1}", value=f"Activo {i+1}")
    capital = st.text_input(f"Capital inicial ($) {i+1}", value="10000")
    rendimiento_esperado = st.text_input(f"Rendimiento anual esperado (%) {i+1}", value="8")
    años = st.number_input(f"Tiempo (años) {i+1}", min_value=1, value=10)
    contribucion = st.text_input(f"Contribución anual ($) {i+1}", value="0")
    
    capital = limpiar_numero(capital)
    rendimiento_esperado = limpiar_numero(rendimiento_esperado)/100
    contribucion = limpiar_numero(contribucion)
    desviacion = rendimiento_esperado * 0.2
    
    activos.append({
        "nombre": nombre,
        "capital": capital,
        "rendimiento_esperado": rendimiento_esperado,
        "años": años,
        "contribucion": contribucion,
        "desviacion": desviacion
    })

if st.button("Ejecutar Simulación"):
    resultados = []
    st.subheader("Gráficas por activo")
    for activo in activos:
        simulaciones = []
        contribuciones_acum = []
        for s in range(num_simulaciones):
            valores = [activo["capital"]]
            contrib_acum = [0]
            for t in range(1, activo["años"]+1):
                r = np.random.normal(activo["rendimiento_esperado"], activo["desviacion"])
                ganancia = valores[-1]*r
                impuestos = ganancia * tasa_impuestos
                nuevo_valor = valores[-1] + ganancia - impuestos + activo["contribucion"]
                valores.append(nuevo_valor)
                contrib_acum.append(contrib_acum[-1] + activo["contribucion"])
            simulaciones.append(valores[1:])
            contribuciones_acum.append(contrib_acum[1:])
        
        simulaciones = np.array(simulaciones)
        finales_reales = simulaciones[:,-1]/((1+inflacion)**activo["años"])
        
        # Estadísticas
        var_5 = np.percentile(finales_reales,5)
        var_1 = np.percentile(finales_reales,1)
        cvar_5 = finales_reales[finales_reales <= var_5].mean()
        cvar_1 = finales_reales[finales_reales <= var_1].mean()
        volatilidad = np.std(finales_reales)
        varianza = np.var(finales_reales)
        skewness = skew(finales_reales)
        kurt_val = kurtosis(finales_reales)
        contrib_total = np.mean(np.array(contribuciones_acum)[:,-1])
        max_drawdown = np.max(np.array([np.max(simulaciones[s]-np.maximum.accumulate(simulaciones[s])) for s in range(num_simulaciones)]))
        sharpe = np.mean(finales_reales)/volatilidad if volatilidad>0 else 0

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
        
        # Gráficas
        promedio = simulaciones.mean(axis=0)
        p5 = np.percentile(simulaciones,5,axis=0)
        p95 = np.percentile(simulaciones,95,axis=0)
        valores_reales_prom = promedio / ((1+inflacion)**np.arange(1, activo["años"]+1))
        
        fig, ax = plt.subplots(1,3, figsize=(18,4))
        ax[0].plot(range(1, activo["años"]+1), valores_reales_prom, marker='o', label="Promedio (real)")
        ax[0].fill_between(range(1, activo["años"]+1),
                           p5/((1+inflacion)**np.arange(1, activo["años"]+1)),
                           p95/((1+inflacion)**np.arange(1, activo["años"]+1)),
                           alpha=0.2, color="orange", label="5%-95%")
        ax[0].set_title("Valor promedio y rango")
        ax[0].grid(True)
        ax[0].legend()
        ax[1].hist(finales_reales, bins=25, alpha=0.7, density=True, color="skyblue")
        ax[1].set_title("Distribución valor final")
        ax[1].grid(True)
        ax[2].boxplot(finales_reales, patch_artist=True)
        ax[2].set_title("Boxplot escenarios finales")
        st.pyplot(fig)
    
    df_resultados = pd.DataFrame(resultados)
    st.subheader("Informe estadístico completo")
    st.dataframe(df_resultados.style.format("${:,.2f}"))
    
    buffer = BytesIO()
    df_resultados.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Descargar informe en Excel",
        data=buffer,
        file_name="informe_portafolio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Mensaje inicial
st.write("👉 Define los parámetros y presiona *Ejecutar Simulación*")
