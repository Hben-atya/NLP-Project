"""

medical VQA project - methods:

1. image encoders: 
  a. VGG 
  b. ResNet (EfficientNet ?)
  c. MOCO - self supervised
  d. VAE 
  
2. language encoders:
  a. BERT
  b. BioBERT
  c. BioMed-RoBERTa-base

3. Fusion Algorithm:
  a. attention mechanism:
    i.   Stacked At-tention Networks (SAN) 
    ii.  Bilinear Attention Networks (BAN)
    iii. transformer?
  b. Multi-modal pooling:
    i.   direct concatenation
    ii.  Multi-modal Compact Bilinear (MCB) pooling
    iii. Multi-modal Factorized Bi-linear (MFB) pooling
  c. MFB pooling with attention solutions

4. Answering Component:
  a. classification mode
  b. generation mode
  c. switching strategy to adopt both classification and genera-tion.


"""
