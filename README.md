# New Phrase Discovery and Keyword Extraction Based on Small-Scale Chinese Text

This repository contains the implementation of a hybrid pipeline designed to **discover new phrases** and **extract keywords** from **limited Chinese datasets** (approximately 100 documents). By combining **traditional statistical methods** with **pre-trained language models**, the system can effectively identify **long phrases** (3-8 characters) and improve downstream keyword extraction performance.

## Core Techniques

Our system employs a multi-stage pipeline to bridge the gap between **traditional statistical modeling** and **modern deep learning**.

### 1. Statistical Candidate Generation

To handle the low frequency of long phrases in small datasets, we use a low-threshold statistical approach to generate a **broad candidate pool**:

* **n-gram & Jieba Merging**: Initial word segmentation and combination.


* **Branch Entropy**: Evaluates word boundary certainty by calculating left and right entropy.


* **Mutual Information**: Uses a custom "aggregation coefficient" ($aggre\_coef$) to determine the correlation between adjacent characters/words.


### 2. Evaluated Filtering Methods

We experimented with two primary neural architectures to filter the statistical candidates:

* **Sequence Labeling (BIO Tagging)**: A model using `chinese-bert-wwm-ext` to generate word embeddings followed by a Linear layer to predict BIO (Begin, Inside, Outside) tags for each character.


* **Textual Entailment (NLI)**: Instead of simple classification, we transform the task into an entailment problem (Natural Language Inference task). Using `chinese-roberta-wwm-ext`, the model evaluates if a sentence entails a specific label template, such as "{} is a phrase about labor".

### 3. Enhanced Keyword Extraction

The final discovered phrases are integrated into a customized dictionary to improve performance of downstream keyword extraction:

* **KeyBERT Weighting**: Keywords matching the discovered dictionary receive a weight multiplier, ensuring domain-specific terminology (e.g., specific labor or entity terms) appear higher in the rankings compared to standard unsupervised methods.

## Experimental Results & Discussion

The model was tested on a self-collected dataset of approximately 200 Chinese news articles and commentaries related to **labor issues**.

| No. | Method | Task Type | F1-Score |
| --- | --- | --- | --- |
| 1 | **BIO Tagging (BERT)** | Sequence Labeling | **0.42** |
| 2 | **Textual Entailment (RoBERTa)** | NLI Template | **93.50** |

Our qualitative analysis shows that combining the discovered dictionary with KeyBERT provides more relevant and domain-specific keywords than **RAKE**, **TextRank**, or **YAKE**.


#### Why did BIO Tagging perform poorly?

The low F1-score (0.42) is mainly due to **severe class imbalance** (the "O" tag vastly outnumbered phrase tags), **inconsistent labels** derived from limited statistical candidates, and the **inherent complexity of sequence labeling**, which requires more annotated data and training resources than the NLI approach and is less suitable for small datasets.
