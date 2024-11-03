# Quantum Walk Implementations
Code to implement continuous and discrete quantum walks on a graph. 
- **Continuous walks** are done on a Hamiltonian set as the Adjacency matrix $$A$$ multiplied by a constant factor, as given in the <a href=https://arxiv.org/pdf/quant-ph/0209131>Childs paper</a>. Evolution is given by $$\ket{\psi(t)} = e^{-iAt}\ket{\psi_0}$$, and $$\ket{\psi_0}$$ is set to $$1$$ on the start node, and 0 on the other ones.
    - In the case with graphs without self-loops, it will be interesting to explore the what happens if we set $$H=A$$ or $$H=L$$, where $$L$$ is the *Laplacian* of the graph. This walk is unique if the nodes are not of equal degree.
- **Discrete walks** are done using a Grover coin to decide the next state to evolve to. Probably not too interesting as it will result in quandratic speedup.

## Codefiles
- <code>walk_visual.ipynb</code> gives an interactive visual on the differences between random and classically-simulated quantum walks. Utilizes matplotlib interactive features to simultaneously compare the random and quantum walks as well as the differences in their probabilities.
- <code>qwalk_simulator.ipynb</code> gives a visual on the differences between random and classically-simulated quantum walks in a printed format. Provides <code>.pdf</code> summary of walks over specified time range.