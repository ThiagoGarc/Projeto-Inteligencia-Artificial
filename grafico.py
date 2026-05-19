#Autores:
#Mateo Zanette RA: 10417980
#Thiago Felipe Garcia RA: 10414699
#Mateus Mendes Cabral RA: 10417820

#Última atualização (19/05):
#-Adição do gráfico "D Chord" como experimentação para a disposição visual das regras



import pandas as pd
import matplotlib.pyplot as plt
from mpl_chord_diagram import chord_diagram

path = r"C:\Users\thiag\OneDrive\Área de Trabalho\Faculdade\Projeto\Regras.csv"


with open(path, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()

header = lines[0].strip().replace('"', '').split(";")
rows = []
for line in lines[1:]:
    values = line.strip().replace('"', '').split(";")
    if len(values) == len(header):
        rows.append(values)

df = pd.DataFrame(rows, columns=header)
df.columns = df.columns.str.strip()
df["confidence"] = df["confidence"].str.replace(",", ".").astype(float)
df["antecedents"] = df["antecedents"].str.strip()
df["consequents"] = df["consequents"].str.strip()

nodes = sorted(set(df["antecedents"]) | set(df["consequents"]))
n = len(nodes)
node_index = {name: i for i, name in enumerate(nodes)}

matrix = [[0.0] * n for _ in range(n)]

for _, row in df.iterrows():
    i = node_index[row["antecedents"]]
    j = node_index[row["consequents"]]
    matrix[i][j] = row["confidence"]

fig, ax = plt.subplots(figsize=(12, 12))
chord_diagram(matrix, nodes, ax=ax, show=False)
plt.tight_layout()
plt.savefig("chord_diagram.png", dpi=150, bbox_inches="tight")
plt.show()