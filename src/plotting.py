import inspect
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import rcParams
from sklearn.manifold import TSNE

# Define a sophisticated color palette
SOPHISTICATED_PALETTE = sns.color_palette("Set2")

# Function to shift color palette
def shift_palette(palette, shift):
    """
    Shifts the color palette by a given offset.

    Parameters:
        palette (list): Original color palette.
        shift (int): Number of positions to shift.

    Returns:
        list: Shifted color palette.
    """
    shift = shift % len(palette)  # Ensure shift wraps around
    return palette[shift:] + palette[:shift]

# Updated font settings
def set_plot_fonts():
    """Sets consistent Times New Roman fonts with slightly larger sizes."""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['legend.title_fontsize'] = 14

# Heatmap Function
def plot_and_save_heatmap(data, folder="Task_1", cmap="viridis", ticks_every=20):
    """
    Plots a heatmap and saves it with the name of the data object.
    """
    set_plot_fonts()  # Set consistent fonts

    # Get the variable name of the passed data object
    caller_locals = inspect.currentframe().f_back.f_locals
    variable_name = [name for name, val in caller_locals.items() if val is data]
    file_name = variable_name[0] if variable_name else "heatmap"

    # Create the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(data, 
                cmap=cmap, 
                square=True, 
                linewidths=0, 
                cbar_kws={"shrink": 0.6}, 
                vmin=0, vmax=1, 
                xticklabels=ticks_every, yticklabels=ticks_every)
    plt.title(f"{file_name}", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the heatmap
    file_path = f"../plots/heatmaps/{folder}/{file_name}_heatmap.png"
    plt.savefig(file_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved as: {file_path}")

# UMAP Function
def plot_and_save_umap(expression_matrix, labels, output_path, 
                       n_neighbors=15, min_dist=0.1, n_components=2, 
                       random_state=42, title="UMAP of Gene Expression",
                       color_shift=False):
    """
    Plots and saves a UMAP projection for the given gene expression data.
    """
    set_plot_fonts()  # Set consistent fonts

    # Perform UMAP
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, random_state=random_state)
    embedding = umap.fit_transform(expression_matrix)
    
    # Create a DataFrame
    umap_df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"], index=expression_matrix.index)
    umap_df['Phenotype'] = labels

    # Shift color palette if enabled
    palette = SOPHISTICATED_PALETTE
    if color_shift:
        palette = shift_palette(SOPHISTICATED_PALETTE, shift=1)

    # Plot UMAP
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="UMAP_1", y="UMAP_2",
        hue="Phenotype", data=umap_df,
        palette=palette, s=80, alpha=0.9, edgecolor=None
    )
    plt.title(title, fontsize=18)
    plt.xlabel("UMAP 1", fontsize=14)
    plt.ylabel("UMAP 2", fontsize=14)
    plt.legend(title="Phenotype", loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"UMAP plot saved to {output_path}")
    plt.show()

# PCA Function
def plot_and_save_pca(expression_matrix, labels, output_path, 
                      n_components=2, random_state=42, title="PCA of Gene Expression",
                      color_shift=False):
    """
    Plots and saves a PCA projection for the given gene expression data.
    """
    set_plot_fonts()  # Set consistent fonts

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(expression_matrix)

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame(pca_result, columns=[f"PCA_{i+1}" for i in range(n_components)], index=expression_matrix.index)
    pca_df['Phenotype'] = labels

    # Shift color palette if enabled
    palette = SOPHISTICATED_PALETTE
    if color_shift:
        palette = shift_palette(SOPHISTICATED_PALETTE, shift=1)

    # Plot PCA
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="PCA_1", y="PCA_2",
        hue="Phenotype", data=pca_df,
        palette=palette, s=80, alpha=0.9, edgecolor=None
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title="Phenotype", loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"PCA plot saved to {output_path}")
    plt.show()

def plot_and_save_tsne(expression_matrix, labels, output_path, 
                       n_components=2, perplexity=30, random_state=42, 
                       title="t-SNE of Gene Expression", color_shift=False):
    """
    Plots and saves a high-quality t-SNE projection for the given gene expression data.

    Parameters:
        expression_matrix (pd.DataFrame): Gene expression matrix.
        labels (pd.Series): Phenotype labels.
        output_path (str): File path to save the t-SNE plot.
        n_components (int): Number of t-SNE components to compute.
        perplexity (int): t-SNE perplexity parameter.
        random_state (int): Random seed.
        title (str): Title for the plot.
        color_shift (bool): If True, shifts the palette by +1.

    Returns:
        None
    """
    set_plot_fonts()  # Set consistent fonts

    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(expression_matrix)

    # Create a DataFrame for visualization
    tsne_df = pd.DataFrame(tsne_result, columns=[f"tSNE_{i+1}" for i in range(n_components)], index=expression_matrix.index)
    tsne_df['Phenotype'] = labels

    # Shift color palette if enabled
    palette = SOPHISTICATED_PALETTE
    if color_shift:
        palette = shift_palette(SOPHISTICATED_PALETTE, shift=1)

    # Map colors to unique labels
    unique_classes = labels.unique()
    class_to_color = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}

    # Plot t-SNE
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="tSNE_1", y="tSNE_2",
        hue="Phenotype", data=tsne_df,
        palette=class_to_color, s=80, alpha=0.9, edgecolor=None
    )
    plt.title(title, fontsize=18)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(title="Phenotype", loc='upper right', frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"t-SNE plot saved to {output_path}")
    plt.show()