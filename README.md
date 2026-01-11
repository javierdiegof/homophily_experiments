
# Understanding the Role of Local Homophily on GNN Performance

  

**Course:** CS512 Data Mining Principles (DMP), Fall 2025

  

**Institution:** University of Illinois Urbana-Champaign (UIUC)

  

**Authors:** Kamila Abdiyeva, Brandon Baez, Javier Diego

  

## Abstract

  

This project systematically evaluates the role of homophily (the tendency of connected nodes to share the same label) at both global and local scales to quantify its effect on node representations and predictive performance in Graph Neural Networks (GNNs). While label homophily is a known predictor of GNN accuracy, this work investigates the mechanistic role of message passing by analyzing embedding dynamics and performing causal perturbations.

  

## Methodology

  

The research is conducted in four interconnected stages:

  

1.  **Global Structural Characterization:** Analyzing global graph statistics (homophily, average degree, clustering coefficient) to relate them to overall GNN performance.

  
  

2.  **Baseline Comparisons:** Disentangling structural versus feature signals by comparing GNNs against:

*  **Node-Label Majority (NLM):** A baseline relying solely on neighborhood labels.

  
  

*  **Feature-only MLP:** A baseline that ignores graph structure entirely.

  
  
  
  

3.  **Representation Dynamics:** Tracking how node embeddings evolve across layers to determine if message passing pulls representations toward homophilous neighbors (embedding drift).

  
  

4.  **Causal Perturbation:** Controlled removal of homophilous neighbors to measure the robustness of GNN predictions and identify the "tipping point" for misclassification.

  
  
  

## Models & Architecture

  

The study evaluates five distinct models using PyG (PyTorch Geometric) standard splits:

  

*  **GCN (Graph Convolutional Network):** Representative spectral architecture.

  
  

*  **GAT (Graph Attention Network):** Attention-based architecture.

  
  

*  **GPRGNN:** A heterophily-robust architecture.

  
  

*  **NLM (Node-Label Majority):** Voting based on 1-hop neighbor labels.

  
  

*  **MLP (Multi-Layer Perceptron):** Feature-only network.

  
  
  

**Hyperparameters:**

  

* Hidden Dimensions: 64.

  
  

* Optimizer: Adam (implied by standard setup, explicitly trained for 200 epochs).

  
  

* Dropout: 0.5 (GCN/GPRGNN) to 0.6 (GAT).

  
  
  

## Datasets

  

The project utilizes 13 benchmark datasets spanning various domains and homophily levels:

  

*  **Homophilous:** Questions, Cora, Coauthor-CS, PubMed, Amazon-Computers, CiteSeer.

  
  

*  **Heterophilous:** Minesweeper, Tolokers, Amazon-ratings, Chameleon, Actor, Squirrel, Roman-empire.

  
  
  

## Key Findings

  

### 1. The Power of Simple Baselines

  

*  **Majority Vote:** A simple label-based majority-vote classifier (NLM) frequently matches or exceeds the accuracy of message-passing GNNs on homophilous datasets. This suggests that local label agreement often explains GNN performance rather than complex non-linear aggregation.

  
  

*  **MLP vs. GNN:** On heterophilous datasets, feature-only MLPs often match or outperform GNNs, indicating that message passing can introduce noise when structural information is not aligned with labels.

  
  
  

### 2. Embedding Drift

  

* Correctly classified nodes undergo a "drift" where their embeddings move closer to the embeddings of their homophilous neighbors across layers.

  
  

* This effect is weak or absent for misclassified nodes or those in heterophilous neighborhoods.

  
  
  

### 3. Sensitivity to Perturbation

  

* GNN predictions are highly sensitive to neighborhood label alignment. In perturbation experiments, predictions collapsed under medium-high relative homophily reductions.

  
  

* GCN predictions generally require large reductions in homophily to flip, but GAT showed higher stability on heterophilous datasets, likely due to its attention mechanism isolating useful neighbors.

  
  
  

## Metrics

  

The paper introduces and utilizes specific metrics for analysis:

  

*  **Message-Passing Drift ():** A directional measure of how message passing pulls a node's representation toward homophilous vs. heterophilous neighbors.

  
  

*  **Local Homophily ():** The proportion of a node's neighbors that share its label.

  
  

*  **Class-wise Cosine Similarity:** Measures how well classes separate in the embedding space after message passing.
