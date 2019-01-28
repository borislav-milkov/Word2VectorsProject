import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv('Thrones2Vec.csv')

ax = df.plot.scatter("x", "y", s=35, figsize=(10, 8))
for i, point in df.iterrows():
    ax.text(point.x, point.y, point.word, fontsize=11)

plt.show()