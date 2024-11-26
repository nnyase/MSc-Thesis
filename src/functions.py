import dcor
import numpy as np
import gudhi as gd
from scipy.stats import pearsonr
from tqdm import tqdm
import math
from gudhi.representations import Landscape

def compute_distance_correlation_matrix(DF):
    
    num_genes = DF.shape[1]
    dist = np.zeros((num_genes, num_genes))
    
    for i in range(num_genes):
        
        for j in range(i+1, num_genes):
            
            dist[i,j] = dcor.distance_correlation(DF[:,i], DF[:,j]) #Distance Correlations 
    
    dist = dist + dist.T + np.eye(num_genes)
    
    return 1 - dist

def compute_pearson_correlation_matrix(DF: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation matrix for the given gene expression data.

    Parameters:
    - DF: np.ndarray, gene expression data with rows as samples and columns as genes.

    Returns:
    - corr_matrix: np.ndarray, the Pearson correlation matrix.
    """
    num_genes = DF.shape[1]
    corr_matrix = np.zeros((num_genes, num_genes))

    for i in range(num_genes):
        for j in range(i, num_genes):
            if i == j:
                corr_matrix[i, j] = 1.0  # Perfect correlation with itself
            else:
                corr, _ = pearsonr(DF[:, i], DF[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetric matrix

    return corr_matrix

def patient_correlation_measure(F, M):
    
    F = F.T
    num_genes = M.shape[1]
    dist = np.zeros((num_genes, num_genes))
        
    for i in range(num_genes):
        for j in range(i+1, num_genes):
            
            dist[i,j] = M[i,j] + math.sqrt(F[i] + F[j])/10 #smaller values are shrunk and larger values are inflated 
    
    dist = dist + dist.T + np.eye(num_genes)
    
    return dist

def compute_wto_matrix(gene_exp_arr: np.ndarray, adj_matrix="dc")-> np.ndarray:
    """
    Compute the Signed Weighted Topological Overlap (wTO) matrix.

    Parameters:
    - df: np.ndarray, input data.

    Returns:
    - wto_matrix: np.ndarray, Signed Topological Overlap matrix
    """
    if adj_matrix == "dc":
        adjacency_matrix = compute_distance_correlation_matrix(DF=gene_exp_arr)
    elif adj_matrix == "pearsons":
        adjacency_matrix = compute_pearson_correlation_matrix(DF=gene_exp_arr)

    num_genes = adjacency_matrix.shape[0]
    wto_matrix = np.zeros((num_genes, num_genes))

    for i in range(num_genes):
        for j in range(num_genes):
            if i != j:
                min_ki_kj = (
                    min(
                        np.sum(np.abs(adjacency_matrix[i, :])),
                        np.sum(np.abs(adjacency_matrix[j, :])),
                    )
                    + 1
                    - np.abs(adjacency_matrix[i, j])
                )
                wto_matrix[i, j] = (
                    np.dot(adjacency_matrix[i, :], adjacency_matrix[:, j])
                    + adjacency_matrix[i, j]
                ) / min_ki_kj

    return wto_matrix


def compute_simplicial_complex_and_landscapes(patient_data: np.ndarray, dist_corr_matrix: np.ndarray, num_landscape=2, resolution=10, dim=2):
    """
    Compute persistent homology and persistence landscapes for all patients.
    
    Parameters:
    - patient_data: np.ndarray, gene expression data with rows as patients and columns as genes.
    - dist_corr_matrix: np.ndarray, population-wide distance correlation matrix.
    - max_filtration: float, maximum filtration value to limit the persistence computation.
    
    Returns:
    - persistence_landscapes: np.ndarray, persistence landscapes for all patients.
    """
    Persistent_diagrams0, Persistent_diagrams1, Persistent_diagrams2 = [], [], []
    print("Computing the Simplicial Complex and persistence for patients")
    for patient_exp in tqdm(patient_data):
        patient_matrix = patient_correlation_measure(patient_exp, dist_corr_matrix)
        #print(patient_matrix.shape)
        rips_complex = gd.RipsComplex(distance_matrix=patient_matrix, max_edge_length=float('Inf')).create_simplex_tree(
                max_dimension=dim)  # Weights used include per-patient gene expressions
        rips_complex.collapse_edges()
        rips_complex.expansion(3)
        diag = rips_complex.persistence()

        Persistent_diagrams0.append(rips_complex.persistence_intervals_in_dimension(0))
        Persistent_diagrams1.append(rips_complex.persistence_intervals_in_dimension(1))
        Persistent_diagrams2.append(rips_complex.persistence_intervals_in_dimension(2))


        #remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])

    #Persistent_diagrams0 = list(map(remove_infinity, Persistent_diagrams0))

    landscape = Landscape(num_landscapes=num_landscape, resolution=resolution)
    samplelandscape0lnd = landscape.fit_transform(Persistent_diagrams0)
    samplelandscape1lnd = landscape.fit_transform(Persistent_diagrams1)
    samplelandscape2lnd = landscape.fit_transform(Persistent_diagrams2)
    landscapes = np.column_stack((samplelandscape0lnd, samplelandscape1lnd, samplelandscape2lnd))

    return landscapes


