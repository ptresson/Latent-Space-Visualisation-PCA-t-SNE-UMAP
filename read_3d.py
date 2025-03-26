from pathlib import Path
obj_path = Path("./data/test.stl")


import numpy as np
from stl import mesh

# Using an existing stl file:
m = mesh.Mesh.from_file(obj_path)
# print(your_mesh.x.shape)
# print(your_mesh.y.shape)
# print(your_mesh.z.shape)

# print(m.points.shape)
# print(m.vectors.shape)
# print(m.vectors)

firsts = m.vectors[:,0,:]
# print(firsts.shape)

subset_size = 5_000
random_indices = np.random.choice(firsts.shape[0], subset_size, replace=False)
subset = firsts[random_indices]
print(subset.shape)
print(subset)

from mpl_toolkits import mplot3d
from matplotlib import pyplot
import seaborn as sns
sns.set_theme(style="whitegrid")


pyplot.figure(figsize=(6, 6))
axes = pyplot.axes(projection="3d")
print(type(axes))
axes.scatter3D(subset[:,0], subset[:,1], subset[:,2], marker='+', alpha=.7)

axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_zlabel("z")

axes.view_init(45, 215)
pyplot.show()
