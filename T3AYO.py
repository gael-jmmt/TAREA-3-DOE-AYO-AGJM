import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


MATRICULA = 2096042
random.seed(MATRICULA)
r = random.random()

T_MIN_RANGO = 0.5
T_MAX_RANGO = 500.0

fracciones = {
    'Bajo': 0.3 * r,
    'Medio': 0.3 * r + 0.3,
    'Alto': 0.3 * r + 0.6
}

niveles_temp = {
    nivel: round(T_MIN_RANGO + frac * (T_MAX_RANGO - T_MIN_RANGO), 4)
    for nivel, frac in fracciones.items()
}


def calcular_matriz_distancias(ciudades):
    """Calcula la matriz de distancias euclidianas entre todas las ciudades."""
    n = len(ciudades)
    return [[math.hypot(ciudades[i][0] - ciudades[j][0], ciudades[i][1] - ciudades[j][1]) for j in range(n)] for i in range(n)]

def costo_ruta(ruta, dist_matrix):
    """Calcula el costo total (distancia) de un tour completo."""
    return sum(dist_matrix[ruta[i-1]][ruta[i]] for i in range(len(ruta)))

def recocido_simulado_doe(dist_matrix, T_inicial, seed_corrida, alpha=0.88, max_iter=120):
    
    rng = random.Random(seed_corrida)
    n = len(dist_matrix)
    
    ruta = list(range(n))
    rng.shuffle(ruta)
    costo_actual = costo_ruta(ruta, dist_matrix)
    
    T = T_inicial
    T_min_corte = 1e-4
    
    for _ in range(max_iter):
        i, j = sorted(rng.sample(range(n), 2))
        
        if j - i == 1 or (i == 0 and j == n - 1):
            T = max(T_min_corte, T * alpha)
            continue
            
        n1, n2 = ruta[i-1], ruta[i]
        n3, n4 = ruta[j-1], ruta[j] if j < n else ruta[0]
        
        delta = (dist_matrix[n1][n3] + dist_matrix[n2][n4]) - (dist_matrix[n1][n2] + dist_matrix[n3][n4])
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            ruta[i:j] = reversed(ruta[i:j])
            costo_actual += delta
            
        T = max(T_min_corte, T * alpha)
        
    return costo_actual


if __name__ == '__main__':
    print("=" * 70)
    print(f"PARÁMETROS INICIALES (Semilla Matrícula: {MATRICULA})")
    print(f"Número aleatorio único r ~ U(0,1): {r:.6f}")
    print(f"Rango de Temperatura Inicial: [{T_MIN_RANGO}, {T_MAX_RANGO}]")
    for k, v in niveles_temp.items():
        print(f"  - Nivel {k:<5}: fracción = {fracciones[k]:.4f}  -->  T0 = {v:.4f}")
    print("=" * 70)

    TAMANOS = [5, 10, 15]
    REPLICAS = 10
    instancias = {}

    for n in TAMANOS:
        rng_inst = random.Random(42 + n)
        ciudades = [(rng_inst.uniform(0, 100), rng_inst.uniform(0, 100)) for _ in range(n)]
        instancias[n] = calcular_matriz_distancias(ciudades)

    resultados = []
    corrida_id = 1

    for n in TAMANOS:
        dist_matrix = instancias[n]
        for nivel_nombre, temp_val in niveles_temp.items():
            for rep in range(1, REPLICAS + 1):
                seed_corrida = MATRICULA + corrida_id * 100 + rep
                costo_final = recocido_simulado_doe(dist_matrix, temp_val, seed_corrida)
                resultados.append({
                    'corrida': corrida_id,
                    'tamano': n,
                    'nivel_temperatura': nivel_nombre,
                    'temperatura': temp_val,
                    'replica': rep,
                    'semilla': seed_corrida,
                    'costo': round(costo_final, 4)
                })
                corrida_id += 1

    df = pd.DataFrame(resultados)
    df.to_csv('resultados_doe_opcionA.csv', index=False)
    print("\nDatos de las 90 corridas exportados a 'resultados_doe_opcionA.csv'.")

    modelo = ols('costo ~ C(temperatura) * C(tamano)', data=df).fit()
    tabla_anova = sm.stats.anova_lm(modelo, typ=2)
    print("\n--- TABLA ANOVA DE DOS FACTORES (IC 95%, alpha = 0.05) ---")
    print(tabla_anova)

    
    np.random.seed(MATRICULA)
    B_BOOT = 10000
    resumen_bootstrap = []

    for n in TAMANOS:
        for nivel_nombre, temp_val in niveles_temp.items():
            datos_grupo = df[(df['tamano'] == n) & (df['temperatura'] == temp_val)]['costo'].values
            medianas_boot = [np.median(np.random.choice(datos_grupo, size=len(datos_grupo), replace=True)) for _ in range(B_BOOT)]
            mediana_obs = np.median(datos_grupo)
            ic_inf, ic_sup = np.percentile(medianas_boot, [2.5, 97.5])
            media_obs = np.mean(datos_grupo)
            
            resumen_bootstrap.append({
                'tamano': n,
                'nivel_temperatura': nivel_nombre,
                'temperatura': temp_val,
                'media': round(media_obs, 2),
                'mediana_bootstrap': round(mediana_obs, 2),
                'ic_95_inf': round(ic_inf, 2),
                'ic_95_sup': round(ic_sup, 2)
            })

    df_boot = pd.DataFrame(resumen_bootstrap)
    print("\n--- MEDIANA BOOTSTRAP E IC 95% POR PUNTO DE DISEÑO ---")
    print(df_boot.to_string(index=False))

    mejores = df_boot.sort_values(by=['tamano', 'mediana_bootstrap', 'media']).groupby('tamano').first().reset_index()
    print("\n--- TABLA RESUMEN: TEMPERATURA ÓPTIMA SELECCIONADA POR TAMAÑO ---")
    print(mejores[['tamano', 'nivel_temperatura', 'temperatura', 'mediana_bootstrap', 'ic_95_inf', 'ic_95_sup']].to_string(index=False))

    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    etiquetas_temp = [f"Bajo\n({niveles_temp['Bajo']:.1f})", 
                      f"Medio\n({niveles_temp['Medio']:.1f})", 
                      f"Alto\n({niveles_temp['Alto']:.1f})"]

    medias_temp = df.groupby('temperatura')['costo'].mean()
    errores_temp = df.groupby('temperatura')['costo'].sem() * 1.96
    axes[0].errorbar(etiquetas_temp, medias_temp.values, yerr=errores_temp.values, 
                     fmt='-o', color='#2980b9', ecolor='#7f8c8d', elinewidth=2, capsize=5, linewidth=2, markersize=8)
    axes[0].set_title('Efecto Principal: Temperatura ($T_0$)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Nivel de Temperatura Inicial ($T_0$)', fontsize=11)
    axes[0].set_ylabel('Costo Promedio de la Ruta', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    medias_tam = df.groupby('tamano')['costo'].mean()
    errores_tam = df.groupby('tamano')['costo'].sem() * 1.96
    etiquetas_tam = [f"n = {n}" for n in TAMANOS]
    axes[1].errorbar(etiquetas_tam, medias_tam.values, yerr=errores_tam.values, 
                     fmt='-s', color='#27ae60', ecolor='#7f8c8d', elinewidth=2, capsize=5, linewidth=2, markersize=8)
    axes[1].set_title('Efecto Principal: Tamaño ($n$)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Tamaño de la Instancia TSP ($n$)', fontsize=11)
    axes[1].set_ylabel('Costo Promedio de la Ruta', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    colores_interaccion = {5: '#e74c3c', 10: '#2980b9', 15: '#8e44ad'}
    marcadores = {5: 'o', 10: 's', 15: '^'}
    for n in TAMANOS:
        sub_df = df[df['tamano'] == n]
        medias_int = sub_df.groupby('temperatura')['costo'].mean()
        errores_int = sub_df.groupby('temperatura')['costo'].sem() * 1.96
        axes[2].errorbar(etiquetas_temp, medias_int.values, yerr=errores_int.values,
                         fmt=f'-{marcadores[n]}', color=colores_interaccion[n],
                         label=f'Tamaño n = {n}', capsize=4, linewidth=2, markersize=7, alpha=0.9)

    axes[2].set_title('Interacción: Temperatura × Tamaño', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Nivel de Temperatura Inicial ($T_0$)', fontsize=11)
    axes[2].set_ylabel('Costo Promedio de la Ruta', fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(title='Bloque (Tamaño)')

    plt.suptitle('Diseño de Experimentos (Opción A: Factorial 3×3) - Recocido Simulado en TSP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grafica_tarea3_doe_opcionA.png', dpi=300, bbox_inches='tight')
    print("\nGráfica guardada como 'grafica_tarea3_doe_opcionA.png'.")
    plt.show()