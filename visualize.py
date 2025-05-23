import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os




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
                ax.set_title(os.path.basename(file_paths[selected_indices[i]]), fontsize=6)
            else:
                ax.set_visible(False)
        
        # Ajouter un titre de ligne (cluster) à gauche
        axes[cluster_id, 0].set_ylabel(f"Cluster {cluster_id}", fontsize=8, rotation=0, labelpad=40, va='center')
    
    plt.tight_layout()
    plt.show()