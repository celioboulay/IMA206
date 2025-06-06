import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import seaborn as sns
import umap.umap_ as umap





def plot_all_clusters_images(file_paths, labels, n_clusters, n_images=5, img_size=(2, 2)):
    fig, axes = plt.subplots(n_clusters, n_images, figsize=(img_size[0]*n_images, img_size[1]*n_clusters))
    
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        selected_indices = np.random.choice(indices, min(len(indices), n_images), replace=False)
        
        for i in range(n_images):
            ax = axes[cluster_id, i] if n_clusters > 1 else axes[i]
            ax.axis('off')
            
            if i < len(selected_indices):
                img = mpimg.imread(file_paths[selected_indices[i]])
                ax.imshow(img)
                ax.set_title(os.path.basename(file_paths[selected_indices[i]]).split('_')[0], fontsize=6)
            else:
                ax.set_visible(False)
        
        # Ajouter un titre de ligne (cluster) à gauche
        axes[cluster_id, 0].set_ylabel(f"Cluster {cluster_id}", fontsize=8, rotation=0, labelpad=40, va='center')
    
    plt.tight_layout()
    plt.show()






def plot_umap(z_3d, painters, epoch, limits):
    unique_painters = sorted(set(painters))
    cmap = cm.get_cmap('tab20', len(unique_painters))
    painter_to_color = {p: cmap(i) for i, p in enumerate(unique_painters)}
    colors = [painter_to_color[p] for p in painters]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], s=50, c=colors, alpha=0.8)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


    #ax.legend(title="Peintres", bbox_to_anchor=(1.05, 1), loc='upper left')
    for p in unique_painters:
        ax.scatter([], [], [], color=painter_to_color[p], label=p, s=50)

    plt.tight_layout()
    plt.savefig(f"output/umap_epoch_{epoch}.png")
    plt.close()





def confusion_print(m):
    plt.figure(figsize=(8, 6))
    sns.heatmap(m, annot=True, fmt='d', cmap='viridis')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title(str(m))
    plt.show()
