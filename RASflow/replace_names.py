import pandas as pd
groups = ['PBS_DONOR', 'VANCO_DONOR', 'PBS_RECIPIENT', 'VANCO_RECIPIENT']

for grp in groups:
    df = pd.read_csv(f'dea/countGroup/{grp}_gene_norm.tsv', sep='\t')
    df.index.name = 'gene_id'
    genes = pd.read_csv('dea/countGroup/tx2gene.tsv', sep='\t').set_index('ensembl_gene_id')
    dc = (genes['external_gene_name']).to_dict()
    gene_names = [dc[x] for x in df.index]
    df['gene_name'] = gene_names
    df.to_csv(f'dea/countGroup/{grp}_genes_norm_named.tsv', sep='\t')

