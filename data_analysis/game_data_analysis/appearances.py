import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../../database/appearances.csv")

# Seleção das colunas numéricas relevantes
numeric_cols = [
    'player_id', 'yellow_cards',
    'red_cards', 'goals',
    'assists', 'minutes_played'
]
df_numeric = df[numeric_cols].copy()


df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')


df_numeric.dropna(inplace=True)


print("Variância das variáveis numéricas:")
print(df_numeric.var())


print("Matriz de Correlação:")
correlation_matrix = df_numeric.corr()
print(correlation_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()
plt.savefig('results/appearances/heatmap.png')


for col in numeric_cols:
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=df_numeric, y=col)
    plt.title(f'Boxplot - {col}')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'results/appearances/boxplot-{col}.png')


df_numeric.hist(bins=10, figsize=(10, 6))
plt.suptitle("Distribuição das variáveis numéricas")
plt.tight_layout()
plt.show()

plt.savefig('results/appearances/histogram.png')