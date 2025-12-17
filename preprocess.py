#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df = pd.read_table("ENC003.HeartLeftVentricle.allelicMethCounts.processed.txt", sep="\t")


df = df.dropna(subset=["Methylation.Difference"])

from pyfaidx import Fasta
ref = Fasta("../hg38.fa")


def get_seq(chrom, pos, window):
    start = pos - window-1
    end = pos + window
    return ref[chrom][start:end].seq.upper()

sequences = []
labels = []
cutoff = 0.3
window = 250
kmer = 6

for _, row in df.iterrows():
    chrom = row['chromosome']
    pos = int(row['end'])
    seq = get_seq(chrom, pos,window = window)
    if seq is None or len(seq) == 0:
      continue

    seq_list = list(seq)
    center_idx = window
    seq_list[center_idx] = row['Allele2']
    seq_modified = ''.join(seq_list)

    kmers = [seq[i:i+kmer] for i in range(len(seq)-kmer+1)]
    kmers_seq = ' '.join(kmers)

    label = 1 if abs(row['Methylation.Difference']) > cutoff else 0

    sequences.append(kmers_seq)
    labels.append(label)


from sklearn.model_selection import train_test_split

out_df = pd.DataFrame({'sequence': sequences, 'label': labels})

train_df, dev_df = train_test_split(
    out_df,
    test_size=0.2,  # 20% dev
    stratify=out_df['label'],
    random_state=42
)

from sklearn.utils import resample

df_positive = train_df[train_df['label'] == 1]
df_negative = train_df[train_df['label'] == 0]

n_samples = min(len(df_positive), len(df_negative))
df_positive_down = resample(df_positive, n_samples=n_samples, replace=False, random_state=42)
df_negative_down = resample(df_negative, n_samples=n_samples, replace=False, random_state=42)

df_balanced = pd.concat([df_positive_down, df_negative_down]).sample(frac=1, random_state=42)  # shuffle

df_balanced.to_csv("ENC003.HeartLeftVentricle/len500/train.csv", index=False)
dev_df.to_csv("ENC003.HeartLeftVentricle/len500/dev.csv",index=False)

