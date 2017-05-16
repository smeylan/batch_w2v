# batch_w2v
This repository allows you to train a set of Word2Vec models with various parameter settings using the `gensim` package.

Rather than accepting command line arguments, `bathch_w2v.py` takes a .ctrl json that specifies the parameter space, input corpus, and output directory. This is copied into the output directory to maintain a record of the arguments that generated a particular dataset.

A .ctrl file must have:  
- inputCorpus: plaintext on which the model is trained  
- outputPath: path to put the trained models  
- parameters:  
  - workers: number of threads used to train the model  
  - window: scope of the context (n words on either side)  
  - mc: filter vocab to words with n or more appearances  
  - sg: #1 is skipgram, 0 is CBOW  
  - neg: number of negative samples. 0 = hierarchical softmax  

Note that all parallelization happens within the `gensim` package; the number of threads is specifed by the parameter `workers`.

Examples are provided for running two datasets, TASA and Wikipedia:  
`w2v_batch.py --ctrl tasa.ctrl`  
`w2v_batch.py --ctrl wikipedia.ctrl`
