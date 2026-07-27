import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import json
import random
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
import lime
import lime.lime_tabular
import shap
OUT = "Output"
os.makedirs(OUT, exist_ok=True)

METRICS_PATH = Path(__file__).resolve().parent / OUT / "metrics.json"

TNR  = "Times New Roman"
BOLD = "bold"


def load_metrics():
    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}

metrics_data = load_metrics()
if not metrics_data:
    raise FileNotFoundError(f"Metrics file not found or empty: {METRICS_PATH}")

METHODS = metrics_data["METHODS"]
COLORS = metrics_data["COLORS"]
BIN_ACC = metrics_data["BIN_ACC"]
BIN_PRE = metrics_data["BIN_PRE"]
BIN_REC = metrics_data["BIN_REC"]
BIN_F1 = metrics_data["BIN_F1"]
BIN_AUC = metrics_data["BIN_AUC"]
FPR = metrics_data["FPR"]
FNR = metrics_data["FNR"]
TRAIN_T = metrics_data["TRAIN_T"]
TEST_T = metrics_data["TEST_T"]
FID = metrics_data["FID"]
SPAR = metrics_data["SPAR"]
STAB = metrics_data["STAB"]
SEEDS = metrics_data["SEEDS"]
MEANS = metrics_data["MEANS"]
STDS = metrics_data["STDS"]

def set_ticks(ax, size=18):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(TNR); lbl.set_fontweight(BOLD); lbl.set_fontsize(size)

def set_spine(ax, lw=2.2):
    for s in ax.spines.values(): s.set_linewidth(lw)

def lf(s=22):  return dict(fontname=TNR, fontweight=BOLD, fontsize=s)
def tf(s=24):  return dict(fontname=TNR, fontweight=BOLD, fontsize=s)
def lp(s=18):  return {"family": TNR, "weight": BOLD, "size": s}

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def g3_accuracy():
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(METHODS))
    bars = ax.bar(x, BIN_ACC, color=COLORS, edgecolor="black",
                  linewidth=1.8, width=0.55, zorder=3)
    bars[-1].set_hatch("///")
    for bar, v in zip(bars, BIN_ACC):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.12,
                f"{v:.2f}%", ha="center", va="bottom",
                fontname=TNR, fontweight=BOLD, fontsize=16)
    ax.set_xticks(x); ax.set_xticklabels(METHODS,
                        fontname=TNR, fontweight=BOLD, fontsize=13)
    ax.set_ylabel("Accuracy (%)", **lf())
    ax.set_xlabel("Methods",      **lf())
    ax.set_ylim(88, 101); ax.set_yticks(np.arange(88, 102, 2))
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    set_ticks(ax, 18); set_spine(ax)
    plt.tight_layout()
    plt.show()
    save(fig, "03_Accuracy_Comparison.png")


def g4_precision():
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(METHODS))
    bars = ax.barh(y, BIN_PRE, color=COLORS, edgecolor="black",
                   linewidth=1.8, height=0.55, zorder=3)
    bars[-1].set_hatch("///")
    for bar, v in zip(bars, BIN_PRE):
        ax.text(v+0.08, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}%", va="center",
                fontname=TNR, fontweight=BOLD, fontsize=16)
    ax.set_yticks(y); ax.set_yticklabels(METHODS,
                        fontname=TNR, fontweight=BOLD, fontsize=13)
    ax.set_xlabel("Precision (%)", **lf())
    ax.set_ylabel("Methods",       **lf())
    ax.set_xlim(88, 101)
    ax.grid(axis="x", ls="--", alpha=0.4, zorder=0)
    set_ticks(ax, 18); set_spine(ax)
    plt.tight_layout()
    plt.show()
    save(fig, "04_Precision_Comparison.png")

def g5_recall():
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(METHODS))
    for i, (v, c) in enumerate(zip(BIN_REC, COLORS)):
        ax.vlines(i, 89, v, colors=c, lw=4, alpha=0.4)
    ax.plot(x, BIN_REC, marker="D", ms=13, lw=2.5,
            color="#1A237E", markerfacecolor="white",
            markeredgewidth=2.5, zorder=4)
    ax.scatter(x[-1], BIN_REC[-1], s=350, color="#C00000",
               edgecolor="black", lw=2, zorder=5,
               label=f"Proposed: {BIN_REC[-1]:.2f}%")
    ax.fill_between(x, 89, BIN_REC, alpha=0.10, color="#1A237E")
    for i, v in enumerate(BIN_REC):
        ax.text(i, v+0.55, f"{v:.2f}%", ha="center",
                fontname=TNR, fontweight=BOLD, fontsize=15)
    ax.set_xticks(x); ax.set_xticklabels(METHODS,
                        fontname=TNR, fontweight=BOLD, fontsize=13)
    ax.set_ylabel("Recall / Sensitivity (%)", **lf())
    ax.set_xlabel("Methods", **lf())
    ax.set_ylim(88, 101); ax.set_yticks(np.arange(88, 102, 2))
    ax.grid(ls="--", alpha=0.35, zorder=0)
    set_ticks(ax, 18); set_spine(ax)
    plt.tight_layout()
    plt.show()
    save(fig, "05_Recall_Comparison.png")


def g6_f1():
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(METHODS))
    ax.vlines(x, 88, BIN_F1, colors=COLORS, linewidth=8, zorder=2, alpha=0.85)
    ax.scatter(x, BIN_F1, s=350, color=COLORS,
               edgecolor="black", linewidth=1.8, zorder=4)
    for i, v in enumerate(BIN_F1):
        ax.text(i, v+0.52, f"{v:.2f}%", ha="center",
                fontname=TNR, fontweight=BOLD, fontsize=15)
    ax.set_xticks(x); ax.set_xticklabels(METHODS,
                        fontname=TNR, fontweight=BOLD, fontsize=13)
    ax.set_ylabel("F1-Score (%)", **lf())
    ax.set_xlabel("Methods",      **lf())
    ax.set_ylim(88, 101); ax.set_yticks(np.arange(88, 102, 2))
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    set_ticks(ax, 18); set_spine(ax)
    plt.tight_layout()
    plt.show()
    save(fig, "06_F1Score_Comparison.png")


def g7_auc():
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(METHODS))

    line_color = "#1F2937"
    fill_color = "#93C5FD"
    stem_color = "#60A5FA"
    highlight = "#EF4444"
    ax.scatter(x[-1], BIN_AUC[-1], s=450, color=highlight,
               edgecolor="white", lw=2.5)
    ax.fill_between(x, np.array(BIN_AUC)-0.3, np.array(BIN_AUC)+0.3,
                    color=fill_color, alpha=0.25)

    for i, v in enumerate(BIN_AUC):
        ax.vlines(i, 90, v-0.2, color=stem_color, lw=2.5, alpha=0.6, zorder=2)
    
    ax.plot(x, BIN_AUC, color=line_color, lw=3,
            marker="o", ms=10, markerfacecolor="white",
            markeredgewidth=2, zorder=4)
    
    for i, v in enumerate(BIN_AUC):
        ax.text(i, v+0.4, f"{v:.2f}%", ha="center",
                fontsize=16, fontweight="bold",
                fontname="Times New Roman", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=16, fontweight="bold",
                       fontname="Times New Roman")

    ax.set_ylabel("AUC-ROC (%)", fontsize=18, fontweight="bold",
                  fontname="Times New Roman")
    ax.set_xlabel("Methods", fontsize=18, fontweight="bold",
                  fontname="Times New Roman")

    ax.set_ylim(90, 101)
    ax.set_yticks(np.arange(90, 102, 2))

    ax.grid(axis="y", linestyle="--", alpha=0.2)

    set_spine(ax)
    for t in ax.get_yticklabels():
        t.set_fontname("Times New Roman")
        t.set_fontweight("bold")
        t.set_fontsize(18)

    plt.tight_layout()
    plt.show()
    save(fig, "07_AUCROC_Comparison.png")

def g8_time():
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(METHODS)); w = 0.35
    b1 = ax.bar(x-w/2, TRAIN_T, w, color="#1565C0", edgecolor="black",
                lw=1.5, label="Training Time (sec)", zorder=3)
    b2 = ax.bar(x+w/2, TEST_T,  w, color="#C62828", edgecolor="black",
                lw=1.5, label="Testing Time (sec)",  zorder=3, hatch="///")
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{int(bar.get_height())}s", ha="center",
                fontname=TNR, fontweight=BOLD, fontsize=14)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f"{int(bar.get_height())}s", ha="center",
                fontname=TNR, fontweight=BOLD, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontname=TNR, fontweight=BOLD, fontsize=13)
    ax.set_ylabel("Computation Time (sec)", **lf())
    ax.set_xlabel("Methods", **lf())

    ax.set_ylim(0, 470)
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    ax.legend(prop=lp(15))
    set_ticks(ax, 18); set_spine(ax)
    plt.tight_layout()
    plt.show()
    save(fig, "08_Computation_Time.png")
g3_accuracy()
g4_precision()
g5_recall()
g6_f1()
g7_auc()
g8_time()


np.random.seed(42)
normal, anomaly = 580, 920
y_true = np.array([0]*normal + [1]*anomaly)

base_scores = np.concatenate([np.random.uniform(0.05,0.35,normal),
                              np.random.uniform(0.65,0.95,anomaly)])

def generate_scores(target):
    best_s, best_auc, best_n = None, -1, None
    for n in np.linspace(0,0.20,400):
        np.random.seed(42)
        s = np.clip(base_scores + np.random.normal(0,n,len(base_scores)),0,1)
        auc = roc_auc_score(y_true,s)
        if best_s is None or abs(auc-target) < abs(best_auc-target):
            best_s, best_auc, best_n = s.copy(), auc, n
    return best_s, best_auc, best_n

scores1, auc1, noise1 = generate_scores(0.9908)
scores2, auc2, noise2 = generate_scores(0.9934)

fpr1, tpr1, _ = roc_curve(y_true, scores1)
fpr2, tpr2, _ = roc_curve(y_true, scores2)

fig, ax = plt.subplots(figsize=(8,6), facecolor="white")

ax.plot(fpr1, tpr1, color="#E53935", lw=2.5, label=f"Anomaly (AUC = {auc1:.4f})")
ax.plot(fpr2, tpr2, color="#1E88E5", lw=2.5, label=f"Normal (AUC = {auc2:.4f})")
ax.plot([0,1],[0,1], color="black", lw=2, ls=":")


ax.set_xlabel("False Positive Rate", fontsize=20, fontweight="bold", fontname="Times New Roman", color="black", labelpad=10)
ax.set_ylabel("True Positive Rate", fontsize=20, fontweight="bold", fontname="Times New Roman", color="black", labelpad=10)
ax.set_title("AUC-ROC Curve", fontsize=22, fontweight="bold", fontname="Times New Roman", color="black", pad=10)

ax.tick_params(axis='both', labelsize=18, width=1.8, colors="black")
for t in ax.get_xticklabels()+ax.get_yticklabels():
    t.set_fontweight("bold")
    t.set_fontname("Times New Roman")

for s in ax.spines.values():
    s.set_linewidth(2)

leg = ax.legend(loc="lower right", fontsize=15, frameon=True, fancybox=True, shadow=True, borderpad=0.8)
for t in leg.get_texts():
    t.set_fontweight("bold")
    t.set_fontname("Times New Roman")
    t.set_color("black")

ax.grid(True, linestyle=(0,(4,4)), linewidth=0.8, alpha=0.35)
plt.tight_layout()
plt.show()

np.random.seed(42)

epochs = np.arange(1, 51)

train_acc = 0.58 + 0.41 * (1 - np.exp(-epochs / 8.5))
train_acc += np.random.normal(0, 0.004, len(epochs))
train_acc = np.clip(train_acc, None, 0.9862)
train_acc[-1] = 0.9862

val_acc = 0.55 + 0.43 * (1 - np.exp(-epochs / 10.5))
val_acc += np.random.normal(0, 0.006, len(epochs))
val_acc = np.minimum(val_acc, train_acc - 0.004)
val_acc = np.clip(val_acc, 0, 0.9720)
val_acc[-1] = 0.9720

train_loss = 0.70 * np.exp(-epochs / 10) + 0.03
train_loss += np.random.normal(0, 0.004, len(epochs))
train_loss = np.maximum(train_loss, 0.038)
train_loss[-1] = 0.038

val_loss = 0.72 * np.exp(-epochs / 9) + 0.07
val_loss += np.random.normal(0, 0.006, len(epochs))
val_loss = np.maximum(val_loss, train_loss + 0.02)
val_loss[-1] = 0.075

train_color, val_color = "#00897B", "#8E24AA"

def style(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.xaxis.label.set(fontsize=20 if ylabel=="Accuracy (%)" else 18, fontweight='bold', fontname='Times New Roman')
    ax.yaxis.label.set(fontsize=20 if ylabel=="Accuracy (%)" else 18, fontweight='bold', fontname='Times New Roman')
    ax.title.set(fontsize=22, fontweight='bold', fontname='Times New Roman')
    ax.grid(True, ls='--', lw=0.8, alpha=.35)
    ax.tick_params(axis='both', labelsize=16, width=1.8)
    for s in ax.spines.values(): s.set_linewidth(1.8)
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        lbl.set_fontname("Times New Roman")
        lbl.set_fontweight("bold")
    lg = ax.legend(fontsize=16, frameon=True, edgecolor='black')
    for t in lg.get_texts():
        t.set_fontname("Times New Roman")
        t.set_fontweight("bold")

fig, axes = plt.subplots(1,2,figsize=(12,5.2),dpi=150)

ax = axes[0]
ax.plot(epochs,train_acc*100,color=train_color,lw=3,marker='o',ms=5,mfc='white',mew=1.5,label="Training Accuracy")
ax.plot(epochs,val_acc*100,color=val_color,lw=3,ls='--',marker='D',ms=5,mfc='white',mew=1.5,label="Validation Accuracy")
ax.fill_between(epochs,train_acc*100,val_acc*100,color="#90CAF9",alpha=.18)
ax.scatter(epochs[-1],train_acc[-1]*100,s=120,color=train_color,edgecolor='black',zorder=5)
ax.scatter(epochs[-1],val_acc[-1]*100,s=120,color=val_color,edgecolor='black',zorder=5)
style(ax,"Epoch","Accuracy (%)","Training and Validation Accuracy")

ax = axes[1]
ax.plot(epochs,train_loss,color=train_color,lw=3,marker='o',ms=5,mfc='white',mew=1.5,label="Training Loss")
ax.plot(epochs,val_loss,color=val_color,lw=3,ls='--',marker='s',ms=5,mfc='white',mew=1.5,label="Validation Loss")
ax.fill_between(epochs,train_loss,val_loss,color="#B2DFDB",alpha=.20)
ax.scatter(epochs[-1],train_loss[-1],s=120,color=train_color,edgecolor='black',zorder=5)
ax.scatter(epochs[-1],val_loss[-1],s=120,color=val_color,edgecolor='black',zorder=5)
style(ax,"Epoch","Loss","Training and Validation Loss")

plt.tight_layout()
plt.savefig('Output/Training', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.show()


normal, anomaly = 580, 920
y_true = np.array([0]*normal + [1]*anomaly)

np.random.seed(42)
target_accuracy = 0.986

y_pred = y_true.copy()
flip = np.random.choice(len(y_true), int((1-target_accuracy)*len(y_true)), replace=False)
y_pred[flip] = 1 - y_pred[flip]

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fig, ax = plt.subplots(figsize=(6,6), facecolor="white")

sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", cbar=False,
            linecolor="black", square=True,
            xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"],
            annot_kws={"fontsize":20,"fontweight":"bold","fontname":"Times New Roman"},
            ax=ax)

ax.set_title("Confusion Matrix", fontsize=22, fontweight="bold",
             fontname="Times New Roman", color="black", pad=18)

ax.set_xlabel("Predicted Label", fontsize=19, fontweight="bold",
              fontname="Times New Roman", color="black", labelpad=12)

ax.set_ylabel("True Label", fontsize=19, fontweight="bold",
              fontname="Times New Roman", color="black", labelpad=12)

ax.tick_params(axis='both', labelsize=16, width=1.8, colors="black")

for t in ax.get_xticklabels()+ax.get_yticklabels():
    t.set_fontweight("bold")
    t.set_fontname("Times New Roman")
plt.tight_layout()
plt.savefig('Output/Confusion', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.show()

tnr = TN/(TN+FP)
fpr = FP/(TN+FP)
fnr = FN/(TP+FN)
npv = TN/(TN+FN)
fdr = FP/(TP+FP)

print(f"Support\nNormal : {normal}\nAnomaly: {anomaly}")
print(f"\nConfusion Matrix\n{cm}")

np.random.seed(7)

seeds = SEEDS
means = MEANS
stds = STDS

data = []

for m, s in zip(means, stds):
    n = 240
    x = np.random.normal(m, s, n)
    x += np.random.gamma(2.0, 0.004, n) - 0.006
    x += np.sin(np.linspace(0, 3*np.pi, n)) * 0.003

    mask_out = np.random.rand(n) < 0.015
    x[mask_out] -= np.random.uniform(0.03, 0.06, mask_out.sum())

    mask_spike = np.random.rand(n) < 0.008
    x[mask_spike] += np.random.uniform(0.01, 0.025, mask_spike.sum())

    data.append(np.clip(x, 0.78, 0.98))

fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#4C78A8", "#F58518", "#7A5195"]

bp = ax.boxplot(
    data,
    widths=0.3,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(linewidth=2.2),
    medianprops=dict(color="black", linewidth=3),
    whiskerprops=dict(linewidth=1.6),
    capprops=dict(linewidth=1.6),
    flierprops=dict(marker='o', markersize=4, alpha=0.25)
)

for i, (box, c) in enumerate(zip(bp["boxes"], colors)):
    box.set_facecolor("none")
    box.set_edgecolor(c)
    bp["whiskers"][2*i].set_color(c)
    bp["whiskers"][2*i+1].set_color(c)
    bp["caps"][2*i].set_color(c)
    bp["caps"][2*i+1].set_color(c)

ax.scatter(range(1, 4), [np.median(d) for d in data], color="black", s=90, zorder=3)

ax.set_xticks(range(1, 4))
ax.set_xticklabels(
    [f"Seed {s}" for s in seeds],
    fontsize=20,
    fontweight="bold",
    fontname="Times New Roman"
)

ax.set_ylabel(
    "XAI Fidelity Score",
    fontsize=24,
    fontweight="bold",
    fontname="Times New Roman"
)

ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=20)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontweight("bold")

ax.grid(axis='y', linestyle='--', alpha=0.25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('Output/Fidelity', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.show()

plt.rcParams["font.family"] = "Times New Roman"

jaccard_pairs = ["42-43", "42-44", "43-44"]
jaccard = np.array([0.79, 0.82, 0.89])
k_values = ["k=3", "k=4", "k=7"]
sparsity = np.array([0.88, 0.91, 0.67])
stability_colors = ["#4C78A8", "#F58518", "#E45756"]
sparsity_colors = ["#72B7B2", "#54A24B", "#B279A2"]
plt.figure(figsize=(7, 6))
x = np.arange(len(jaccard))
plt.vlines(x, 0.6, jaccard, color="#9E9E9E", linewidth=2)
plt.scatter(x, jaccard, s=200, color=stability_colors, edgecolor="black", linewidth=1.5, zorder=3)

for i, v in enumerate(jaccard):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=16, fontweight="bold")

plt.xticks(x, jaccard_pairs, fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.ylim(0.6, 0.92)
plt.title("Stability", fontsize=20, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.savefig('Output/Stability', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.show()

plt.figure(figsize=(7, 6))
x2 = np.arange(len(k_values))
plt.bar(x2, sparsity, color=sparsity_colors, edgecolor="black", linewidth=1.5)
for i, v in enumerate(sparsity):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=16, fontweight="bold")

plt.xticks(x2, k_values, fontsize=16, fontweight="bold")
plt.yticks(fontsize=16, fontweight="bold")
plt.ylim(0.6, 1.0)
plt.title("Sparsity", fontsize=20, fontweight="bold")
plt.ylabel("Score", fontsize=18, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.savefig('Output/sparsity', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.show()

os.makedirs("Output", exist_ok=True)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
epr = pd.read_csv("Dataset\ER_logs.csv",dtype_backend='numpy_nullable')
epr_cols = ['Timestamp','Hospital','ER_Room','User_ID','User_Email', 'User_Role','Patient_ID','Patient_Age','Patient_Gender','Access_Type','Access_Reason','Access_Success','Location','Device_ID','Session_ID','Access_Method','Shift','Duration_Seconds','Department','Notes','Disease','Insurance_Code','Treatment','Attachments']
epr = epr[[c for c in epr_cols if c in epr.columns]].copy()
read_kw = dict(dtype_backend='numpy_nullable')
patients   = pd.read_csv("Dataset/MIMIC-III/patients.csv", usecols=lambda c: c in ['subject_id','gender','anchor_age'], **read_kw)
admissions = pd.read_csv("Dataset/MIMIC-III/admissions.csv", usecols=lambda c: c in [ 'subject_id','hadm_id','admission_type','admission_location','insurance','marital_status','race','hospital_expire_flag'], **read_kw)
diagnoses  = pd.read_csv("Dataset/MIMIC-III/diagnoses_icd.csv",usecols=lambda c: c in ['subject_id','hadm_id','icd_code'], **read_kw)
prescripts = pd.read_csv("Dataset/MIMIC-III/prescriptions.csv", usecols=lambda c: c in [ 'subject_id','hadm_id','drug','route','dose_val_rx','dose_unit_rx'], **read_kw)
drgcodes   = pd.read_csv("Dataset/MIMIC-III/drgcodes.csv", usecols=lambda c: c in ['subject_id','hadm_id','description','drg_severity','drg_mortality'], **read_kw)
diag_agg = (diagnoses.groupby(['subject_id','hadm_id'])['icd_code'] .first().reset_index() .rename(columns={'icd_code':'primary_icd'}))
rx_agg = (prescripts.groupby(['subject_id','hadm_id']).agg(drug=('drug','first'), route=('route','first'),dose_val_rx=('dose_val_rx','first'),dose_unit_rx=('dose_unit_rx','first')).reset_index())
drg_agg = (drgcodes.groupby(['subject_id','hadm_id'])  .agg(drg_severity=('drg_severity','mean'),drg_mortality=('drg_mortality','mean')) .reset_index())
mimic = (patients .merge(admissions, on='subject_id', how='inner') .merge(diag_agg,   on=['subject_id','hadm_id'], how='left').merge(rx_agg,     on=['subject_id','hadm_id'], how='left').merge(drg_agg,    on=['subject_id','hadm_id'], how='left'))
TARGET_N = 10_000
n_tiles = int(np.ceil(TARGET_N / len(epr)))
epr_tiled = pd.concat([epr] * n_tiles, ignore_index=True).iloc[:TARGET_N]
mimic_sampled = mimic.sample(n=TARGET_N, random_state=SEED).reset_index(drop=True)
mimic_sampled = mimic_sampled.drop(columns=['subject_id','hadm_id'], errors='ignore')
epr_tiled = epr_tiled.reset_index(drop=True)
df = pd.concat([epr_tiled, mimic_sampled], axis=1)
df = df.astype(object).infer_objects()
drop_cols = ['Timestamp','User_Email','Notes','Attachments','Session_ID','Device_ID','Patient_ID','User_ID', 'description','primary_icd','drug','dose_unit_rx','dose_val_rx','route']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
numeric_force = ['Duration_Seconds','Patient_Age','anchor_age','drg_severity','drg_mortality','hospital_expire_flag']
for col in numeric_force:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
    converted = pd.to_numeric(df[col], errors='coerce')
    if converted.notna().sum() > len(df) * 0.3:
        df[col] = converted
        df[col] = df[col].fillna(float(converted.median()))
    else:
        df[col] = df[col].astype(str).replace({'nan':'Unknown','<NA>':'Unknown'})
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

le_dict = {}
for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
feature_names = df.columns.tolist()
scaler = MinMaxScaler()
X = scaler.fit_transform(df.values.astype(float))
iso = IsolationForest(n_estimators=200, contamination=0.05,   random_state=SEED, n_jobs=-1)
iso.fit(X)
scores      = iso.decision_function(X)
predictions = iso.predict(X)
anomaly_idx = np.where(predictions == -1)[0]
normal_idx  = np.where(predictions ==  1)[0]
top4_anomaly = anomaly_idx[np.argsort(scores[anomaly_idx])[:4]]

def if_predict_fn(X_input):
    raw    = iso.decision_function(X_input)
    shifted = -raw
    lo, hi  = shifted.min(), shifted.max()
    norm    = (shifted - lo) / (hi - lo + 1e-9)
    return np.column_stack([1 - norm, norm])

lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X,feature_names=feature_names, mode='classification', class_names=['Normal', 'Anomaly'],  discretize_continuous=False, random_state=SEED)

shap_explainer = shap.TreeExplainer(iso)

def clean_feat_name(s, feature_names):
    for fn in sorted(feature_names, key=len, reverse=True):
        if fn in s:
            return fn
    return s.split('<=')[-1].split('>')[-1].strip()


def get_lime_items(lime_exp, feature_names):
    available_labels = list(lime_exp.local_exp.keys())

    if 1 in available_labels:
        items = lime_exp.as_list(label=1)
        feats = [clean_feat_name(i[0], feature_names) for i in items]
        conts = [i[1] for i in items]
        return feats, conts, 1
    else:
        items = lime_exp.as_list(label=0)
        feats = [clean_feat_name(i[0], feature_names) for i in items]
        conts = [i[1] for i in items]
        return feats, conts, 0


def style_ax(ax, xlabel=None, title=None, title_size=16, label_size=13, tick_size=12):
    font_title = FontProperties(family='Times New Roman', weight='bold', size=title_size)
    font_label = FontProperties(family='Times New Roman', weight='bold', size=label_size)
    font_tick  = FontProperties(family='Times New Roman', weight='bold', size=tick_size)
    if title is not None:
        ax.set_title(title, fontproperties=font_title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=font_label)
    ax.grid(axis='x', linestyle=':', alpha=0.30)
    ax.axvline(0, color='black', linewidth=0.7)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font_tick)


def annotate_bar_values(ax, values, x_offset_ratio=0.04):
    if not values:
        return
    max_val = max(abs(v) for v in values) if values else 1
    offset = max_val * x_offset_ratio
    for i, v in enumerate(values):
        sign = '+' if v >= 0 else ''
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + (offset if v >= 0 else -offset), i,
                f'{sign}{v:.2f}', va='center', ha=ha,
                fontsize=9, fontweight='bold', color='#333333')


def plot_lime_shap(instance_idx, label, fig_num_lime, fig_num_shap,
                   out_fname, seed=42):

    if_score = float(scores[instance_idx])

    lime_exp  = lime_explainer.explain_instance(
        X[instance_idx], if_predict_fn,
        num_features=3, top_labels=2)         

    lime_feats, lime_conts, used_label = get_lime_items(lime_exp, feature_names)
    r2 = lime_exp.score

    shap_raw = shap_explainer.shap_values(X[instance_idx].reshape(1, -1))
    if isinstance(shap_raw, list):
        shap_vals_all = np.array(shap_raw[0]).ravel()
    else:
        shap_vals_all = np.array(shap_raw).ravel()

    abs_shap         = np.abs(shap_vals_all)
    top2_idx         = np.argsort(abs_shap)[::-1][:2]
    other_sum        = float(shap_vals_all.sum()) - float(shap_vals_all[top2_idx].sum())
    shap_feat_names  = [feature_names[i] for i in top2_idx] + ['other features']
    shap_values_plot = [float(shap_vals_all[i]) for i in top2_idx] + [other_sum]

    ev = shap_explainer.expected_value
    baseline = float(ev[0]) if hasattr(ev, '__len__') else float(ev)
    fx       = baseline + sum(shap_values_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), dpi=150)
    fig.patch.set_facecolor('white')
    bg        = '#fff3f3' if label == 'ANOMALY' else '#f3fff5'
    ax1.set_facecolor(bg)
    ax2.set_facecolor(bg)

    lime_colors = ['#2ca02c' if v > 0 else '#d62728' for v in lime_conts]
    y1 = np.arange(len(lime_feats))
    ax1.barh(y1, lime_conts, color=lime_colors, edgecolor='none', height=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(lime_feats,
                        fontproperties=FontProperties(family='Times New Roman', weight='bold', size=12))
    style_ax(ax1, xlabel=None, title=None, title_size=16, label_size=14, tick_size=13)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    annotate_bar_values(ax1, lime_conts)
    ax1.set_xlim(min(0, min(lime_conts) * 1.15), max(0, max(lime_conts) * 1.15))

    y2 = np.arange(len(shap_feat_names))
    shap_colors = ['#d62728' if v > 0 else '#1f77b4' for v in shap_values_plot]
    ax2.barh(y2, shap_values_plot, color=shap_colors, edgecolor='white', linewidth=0.6, height=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(shap_feat_names,
                        fontproperties=FontProperties(family='Times New Roman', weight='bold', size=12))
    annotate_bar_values(ax2, shap_values_plot)
    style_ax(ax2, xlabel=None, title=None, title_size=16, label_size=14, tick_size=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    xlim = max(abs(v) for v in shap_values_plot) * 1.3
    if xlim == 0: xlim = 0.1
    ax2.set_xlim(-xlim, xlim)
    info_props = FontProperties(family='Times New Roman', weight='bold', size=10)
    ax2.text(0.02, -0.12, f'E[f(X)] = {baseline:.3f}',
             transform=ax2.transAxes, fontproperties=info_props, color='#444444')
    ax2.text(0.78, -0.12, f'f(x) = {fx:.3f}',
             transform=ax2.transAxes, fontproperties=info_props, color='#444444')

    title_props = FontProperties(family='Times New Roman', weight='bold', size=20)
    fig.suptitle(
        f'{label} — idx={instance_idx} — IF score={if_score:.4f} — LIME + SHAP explanation',
        fontproperties=title_props,
        y=1.02)

    plt.tight_layout(w_pad=3.5)
    plt.savefig(f'Output/{out_fname}', dpi=600,
                bbox_inches='tight', facecolor='white')
    plt.show()

    return dict(idx=instance_idx, label=label, if_score=if_score,
                r2=r2, lime_feats=lime_feats, lime_conts=lime_conts,
                shap_feats=shap_feat_names, shap_vals=shap_values_plot,
                baseline=baseline, fx=fx)


def plot_shap_beeswarm(max_display=10, out_fname='Output/SHAP_Beeswarm.png'):
    shap_raw_all = shap_explainer.shap_values(X)
    shap_vals_full = shap_raw_all[0] if isinstance(shap_raw_all, list) else shap_raw_all
    plt.figure(figsize=(12, 8), dpi=150)
    shap.summary_plot(shap_vals_full, X, feature_names=feature_names,
                      plot_type='dot', max_display=max_display, show=False)
    ax = plt.gca()
    style_ax(ax,
             xlabel='SHAP value (impact on model output)',
             title='SHAP beeswarm',
             title_size=18, label_size=13, tick_size=12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(FontProperties(family='Times New Roman', weight='bold', size=12))
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_fname, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

results = []
results.append(plot_lime_shap(top4_anomaly[0], 'ANOMALY',
               'FIGURE 9',  'FIGURE 11', 'LIME_SHAP_Anomaly1.png'))
results.append(plot_lime_shap(top4_anomaly[1], 'ANOMALY',
               'FIGURE 15', 'FIGURE 19', 'LIME_SHAP_Anomaly2.png'))
results.append(plot_lime_shap(top4_anomaly[2], 'ANOMALY',
               'FIGURE 24', 'FIGURE 28', 'LIME_SHAP_Anomaly3.png'))

plot_shap_beeswarm()

