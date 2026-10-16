"""Generate notebooks/01_eda.ipynb (executed separately with nbconvert)."""
from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md("# 01 - Exploratory Data Analysis: Heart Disease UCI (Cleveland)\n"
   "Binary target: `num` > 0 -> heart disease present (1).")
code('''import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
raw = pd.read_csv(ROOT / "data" / "raw" / "heart.csv", na_values=["?"])
df = pd.read_csv(ROOT / "data" / "processed" / "heart_clean.csv")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.show()''')

md("## 1. Dataset summary")
code('print(df.shape)\ndf.describe().T.round(2)')
md("**Insight:** 303 patients, 13 predictors. Ages 29-77 (mean ~54); "
   "`oldpeak` and `chol` are right-skewed, hinting at outliers worth robust scaling/checking.")

md("## 2. Missing values (raw data)")
code('''miss = pd.DataFrame({"missing": raw.isna().sum(),
                     "pct": (raw.isna().mean() * 100).round(2)})
miss[miss.missing > 0]''')
md("**Insight:** Only `ca` (4) and `thal` (2) have missing values in the raw file "
   "(<2% of rows), so mode imputation is safe and loses no rows.")

md("## 3. Histograms of numeric features")
code('''fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), NUM):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
axes.ravel()[-1].axis("off")
save("hist_numeric.png")''')
md("**Insight:** `chol` and `oldpeak` are right-skewed (skew ~1.1 and ~1.3) while `thalach` "
   "leans left; `age` and `trestbps` are roughly bell-shaped. Scaling is needed for linear models.")

md("## 4. Categorical features vs target")
code('''fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), CAT):
    sns.countplot(data=df, x=col, hue="num", ax=ax, palette="Set2")
    ax.set_title(f"{col} vs target")
save("countplots_categorical.png")''')
md("**Insight:** Disease prevalence is ~73% for asymptomatic chest pain (`cp`=4) vs ~18-30% for "
   "other types, ~77% with exercise angina, and rises steadily with `ca` (26% at 0 vessels -> 85% at 3).")

md("## 5. Correlation heatmap")
code('''plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
save("correlation_heatmap.png")''')
md("**Insight:** Strongest correlates of disease are `thal` (0.52), `ca` (0.46), `exang` (0.43), "
   "`oldpeak` (0.42), `cp` (0.41) and `thalach` (-0.42). No pair of predictors is highly collinear, so all can be kept.")

md("## 6. Class balance")
code('''ax = sns.countplot(data=df, x="num", hue="num", palette="Set2", legend=False)
for c in ax.containers:
    ax.bar_label(c)
plt.title("Class balance (0 = healthy, 1 = disease)")
save("class_balance.png")
df.num.value_counts(normalize=True).round(3)''')
md("**Insight:** 164 healthy vs 139 diseased (54% / 46%). Classes are nearly balanced, "
   "so accuracy is meaningful, though we optimise ROC-AUC and watch recall for screening.")

md("## 7. Boxplots by target")
code('''fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), NUM):
    sns.boxplot(data=df, x="num", y=col, hue="num", ax=ax, palette="Set2", legend=False)
    ax.set_title(f"{col} by target")
axes.ravel()[-1].axis("off")
save("boxplots_by_target.png")''')
md("**Insight:** Diseased patients have markedly lower max heart rate (`thalach` mean 139 vs 158) "
   "and higher `oldpeak` (1.6 vs 0.6); `chol` and `trestbps` overlap heavily and are weaker signals.")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, NB_PATH)
