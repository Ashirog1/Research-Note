https://arxiv.org/pdf/2310.15543.pdf

# 1. Intro

Problem: 2D Euclidean space

Incorporate a geometric deep learning architecture to enforce invariance (i.e. symmetry preservation) with respect to rotations, reflections, and translations

# 2. Problem

2D Eculid TSP.

Each node have a location in 2D space. Find TSP tour with minimum distance.

# 3. Method:

![image](https://github.com/Ashirog1/Research-Note/assets/71763648/1c7351ee-6b46-433f-9191-911a99e8ff23)


## 3.1. Multiresolution graph
Multiresolution graph training for learning routing problems at multiple levels, sub-graphs and high-level graphs.

L-level multiresolution:

$G_l$ is graph created by clustering $G_{l+1}$. With $G_l = G$

sub-instance graph: $G^{sub}_k$ (graph obtained by node inside cluster after clustering G, obtained k subgraph).

high-lvel instance: $G^{high}_l$ (graph obtained by centroid of clustering).

(sub intansce) $G^{sub}_k$ is obtained by clustering 1 time. High-level achived by doing clustering $l$ times.

3.2. Loss function for RL

$L_{total} = L_{orignial} + L_{sub} + L_{high}$.

![image](https://github.com/Ashirog1/Research-Note/assets/71763648/74d6841d-4b03-45a3-b9e6-27102085d632)


3.3. EQUIVARIANT ATTENTION MODEL

Encoder: Equivariant Graph Attention Encoder

input: feature $x_i$ of node $v_i$ (dx = 2). pass through linear layer. Feed them to N equivariant attention layers.

obtain graph embedding = mean of node embedding. $\overline{\rm h} ^ {N} = mean (h_i)$.

Decoder: Create a tour sequentially.



