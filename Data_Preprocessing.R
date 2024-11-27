# Load necessary libraries
library(DESeq2)

# Load Data
counts_matrix <- read.csv("~/Desktop/MSc-Thesis/data/cancer_type/counts_matrix.csv")
conditions <- factor(counts_matrix[['phenotype']])
phenotypes <- counts_matrix$phenotype
table(counts_matrix$phenotype)
counts_matrix$phenotype <- NULL

sample_ids <- counts_matrix$X

# Remove the 'X column from the counts matrix
counts_matrix$X <- NULL

counts <- t(counts_matrix)

sample_info <- data.frame(
  row.names = colnames(counts),
  condition = phenotypes
)

# Initial number of samples and genes
initial_samples <- ncol(counts)  # Number of columns (samples) in the count matrix
initial_genes <- nrow(counts)    # Number of rows (genes) in the count matrix

# Create DESeq2 object
dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = sample_info,
                              design = ~ condition)

# 1A. Estimate dispersions and detect outliers with Cooks distance
dds <- estimateSizeFactors(dds)      # Step 1: Normalize the counts

# Filter low-quality samples (1A)
keep_samples <- colSums(counts(dds) >= 10) > 500  # Keep samples with at least 500 genes expressed above 10
filtered_samples <- sum(!keep_samples)      
dds <- dds[, keep_samples]                       # Subset the dataset to keep only high-quality samples

# Filter low-quality genes (1C)
keep_genes <- rowSums(counts(dds) >= 10) >= 500    # Keep genes expressed in at least 5 samples
filtered_genes <- sum(!keep_genes)          
dds <- dds[keep_genes, ]                         # Subset the dataset to keep only high-quality genes

# Print the number of filtered samples and genes
cat("Number of samples removed (1B):", filtered_samples, "\n")
cat("Number of genes removed (1C):", filtered_genes, "\n")


dds <- estimateDispersions(dds)      # Step 2: Estimate dispersions
dds <- nbinomWaldTest(dds)           # Step 3: Perform the Wald test

dds_lrt <- DESeq(dds, test = "LRT", reduced = ~ 1, minReplicatesForReplace = 7)

res_lrt <- results(dds_lrt)

res_lrt$padj <- p.adjust(res_lrt$pvalue, method = "BH")

alpha <- 0.05 # Adjusted p-value threshold
log2fc_cutoff <- 3  # Log2 fold change threshold

sig_genes <- res_lrt[which(res_lrt$padj < alpha & abs(res_lrt$log2FoldChange) > log2fc_cutoff), ]


EnhancedVolcano(
  res_lrt,
  lab = rownames(res_lrt),  
  x = 'log2FoldChange',     
  y = 'padj',              
  subtitle = 'Significant Genes in LRT Analysis')

vsd <- vst(dds_lrt, blind = FALSE)
pca_data <- plotPCA(vsd, intgroup = "condition", returnData = TRUE)

# Inspect PCA data
head(pca_data)

# Percentage of variance explained
percentVar <- round(100 * attr(pca_data, "percentVar"))

# Replace long labels with shorter ones
pca_data$condition <- gsub("TCGA-ESCA", "ESCA", pca_data$condition)
pca_data$condition <- gsub("TCGA-PCPG", "PCPG", pca_data$condition)
pca_data$condition <- gsub("TCGA-SARC", "SARC", pca_data$condition)


ggplot(pca_data, aes(PC1, PC2, color = condition)) +
  geom_point(size = 3, alpha = 0.8) +                 # Larger points for clarity
  theme_classic(base_size = 12) +                    # Set smaller base size
  labs(
    x = paste0("PC1 (", percentVar[1], "%)"),        # X-axis with variance explained
    y = paste0("PC2 (", percentVar[2], "%)"),        # Y-axis with variance explained
    color = "Condition"                              # Legend title
  )  +  # Muted colors
  theme(
    text = element_text(family = "Times"),        
    axis.title = element_text(size = 8),           # Smaller axis titles
    axis.text = element_text(size = 8),            # Smaller axis text
    legend.position = "top",                        # Move legend to top
    legend.title = element_text(size = 8),         # Smaller legend title
    legend.text = element_text(size = 8),
    axis.line = element_blank(),# Smaller legend text
    legend.direction = "horizontal", 
    # Arrange legend horizontally
    panel.border = element_rect(color = "black",     # Add black rectangular border
                                fill = NA, size = 0.3),
    axis.ticks = element_line(size = 0.3),           # Thinner tick lines
    axis.ticks.length = unit(0.15, "cm") , 
    legend.margin = margin(t = -5, unit = "pt"),
    plot.margin = margin(t = 10, r = 10, b = 10, l = 10)
  )
ggsave("pca_plot_lowered_legend.pdf", width = 6, height = 8, dpi = 600)


plotMA(
  res_lrt,                # DESeq2 results object        # Y-axis limits
  main = "MA Plot",       # Title for the plot
  xlab = "Average log-expression",  # X-axis label
  ylab = "log2-fold-change"         # Y-axis label
)

library(pheatmap)

# Top 50 significant genes
top_sig_genes <- rownames(sig_genes[order(sig_genes$padj), ])[1:100]
vsd <- vst(dds, blind = FALSE)
pheatmap(assay(vsd)[top_sig_genes, ],
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         show_rownames = TRUE,
         scale = "row",
         annotation_col = as.data.frame(colData(dds)["condition"]),
         main = "Heatmap of Top Significant Genes")

plotDispEsts(dds)
write.csv(as.data.frame(sig_genes), "~/Desktop/MSc-Thesis/significant_genes.csv", row.names = TRUE)
