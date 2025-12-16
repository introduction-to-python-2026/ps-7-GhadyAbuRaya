import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.head())
print(df.describe())
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Correlation between Sepal Length and Petal Length")
plt.savefig("correlation.png")
plt.show()

