import sys
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
     

mammoth = pd.read_csv('mammoth_a.csv')
     

print(len(mammoth))
mammoth2 = mammoth.sample(50000)
del mammoth
mammoth = mammoth2

colors = np.sqrt(mammoth['y']**2 + mammoth['z']**2)  # Color points based on their position
print(colors.shape)

pca = PCA(n_components=2)

embeddings = pca.fit_transform(mammoth)
individual_fig, individual_ax = plt.subplots(figsize=(10, 10), facecolor='w')
individual_ax.axis('off')
individual_ax.scatter(embeddings[:, 0], embeddings[:, 1], s=1, c=colors, cmap='gnuplot')
individual_fig.savefig(f'pca.png')
plt.close(individual_fig)  # Close the individual figure to free memory
sys.exit(1)




### 3D view
fig = plt.figure(figsize=(25,25))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1.2, 0.9), zoom=1.5)
ax.set_axis_off()
ax.scatter(mammoth['x'], mammoth['y'], mammoth['z'],s=20,c=colors,cmap='gnuplot')
ax.view_init(0, -170)
plt.tight_layout()
plt.savefig('3d_color.png', bbox_inches='tight')

### 3D view
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1.2, 0.9), zoom=1.5)
ax.set_axis_off()
ax.scatter(mammoth['x'], mammoth['y'], mammoth['z'],s=20,c='black')
ax.view_init(0, -170)
plt.tight_layout()
plt.savefig('3d_single.png', bbox_inches='tight')


### double 3D view
fig = plt.figure(figsize=(50,25))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_box_aspect((1, 1.2, 0.9), zoom=1.5)
ax.set_aspect('equal')
ax.set_axis_off()
ax.scatter(mammoth['x'], mammoth['y'], mammoth['z'], s=20,c=colors,cmap='gnuplot')
ax.view_init(10, -170)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_box_aspect((1, 1.2, 0.9), zoom=1.5)
ax.set_axis_off()
ax.scatter(mammoth['x'], mammoth['y'], mammoth['z'], s=20,c=colors,cmap='gnuplot')
ax.view_init(90, 0)

plt.tight_layout()
plt.savefig('3d_both.png')


sys.exit(1)

params = [5, 7, 9, 10, 50, 100, 500, 1000]

## fit umaps 
for param in params:

    reducer = umap.UMAP(n_neighbors=param,
                        n_components=2,
                        verbose=True,
                        )
    embeddings = reducer.fit_transform(mammoth)
    emb_col =  np.column_stack((embeddings, colors))
    np.save(f'n{param}.npy', emb_col)
print('pouet')


# Create a figure with 6 subplots in a 2x3 grid
fig, axes = plt.subplots(2, 4, figsize=(40, 20), facecolor='w')

# Flatten the axes array for easy iteration
axes = axes.flatten()

for ax, param in zip(axes, params):

    embeddings = np.load(f'n{param}.npy')
    print(embeddings.shape)
    # sys.exit(1)
    
    ax.axis('off')
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], s=2.5, c=embeddings[:,2], cmap='gnuplot')
    ax.set_title(f'n = {param}')

    # Save each subplot as a separate figure
    individual_fig, individual_ax = plt.subplots(figsize=(10, 10), facecolor='w')
    individual_ax.axis('off')
    individual_ax.scatter(embeddings[:, 0], embeddings[:, 1], s=1, c=embeddings[:,2], cmap='gnuplot')
    individual_fig.savefig(f'n{param}.png')
    plt.close(individual_fig)  # Close the individual figure to free memory

# Adjust layout for the combined figure
plt.tight_layout()

# Save the combined figure
plt.savefig('combined_scatter_plots.png')


