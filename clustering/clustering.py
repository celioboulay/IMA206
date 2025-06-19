from sklearn.mixture import BayesianGaussianMixture
import os
import torch
import numpy as np
from collections import defaultdict
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def cluster(embeddings):
    bgmm = BayesianGaussianMixture(n_components=10, covariance_type='full', random_state=42)
    bgmm.fit(embeddings)
    return bgmm

if __name__ == "__main__":
    emb_dir = "embeddings/local/info_vae_dim64_reg10_kld1e-3"
    embeddings = []
    painters = []

    for fname in os.listdir(emb_dir):
        if fname.endswith(".pt"):
            emb = torch.load(os.path.join(emb_dir, fname), weights_only=True)
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            painter = fname.split("_")[0]
            painters.append(painter)
            embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    bgmm = cluster(embeddings)
    painting_probs = bgmm.predict_proba(embeddings)

    painter_probs = defaultdict(list)
    for painter, probs in zip(painters, painting_probs):
        painter_probs[painter].append(probs)

    painter_class_probs = {
        painter: np.mean(probs, axis=0)
        for painter, probs in painter_probs.items()
    }

    # Affichage console
    for painter, probs in painter_class_probs.items():
        print(f"{painter}: {np.round(probs, 3)}")

    # DataFrame pour seaborn
    df = pd.DataFrame.from_dict(painter_class_probs, orient='index')
    df.columns = [f"Class {i}" for i in range(df.shape[1])]
    df.index.name = "Painter"

    # Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Mean class probability per painter (Heatmap)")
    plt.tight_layout()
    plt.show()

    # Barplots par peintre
    n_painters = len(df)
    fig, axes = plt.subplots(n_painters, 1, figsize=(10, 3 * n_painters), sharex=True)

    if n_painters == 1:
        axes = [axes]

    for i, (painter, row) in enumerate(df.iterrows()):
        sns.barplot(x=row.index, y=row.values, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Painter: {painter}")
        axes[i].set_ylabel("Probability")
        axes[i].set_ylim(0, 1)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
