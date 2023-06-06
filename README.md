# The Emergence of Essential Sparsity in\\Large Pre-trained Models: The Weights that Matter

### Abstract
Large pre-trained transformers are show-stealer in modern-day deep learning, and
it becomes crucial to comprehend the parsimonious patterns that exist within them
as they grow in scale. With exploding parameter counts, Lottery Ticket Hypothesis
(LTH) and its variants, have lost their pragmatism in sparsifying them due to high
computation and memory bottleneck of repetitive train-prune-retrain routine of
iterative magnitude pruning (IMP) which worsens with increasing model size. In
this paper, we comprehensively study induced sparse patterns across multiple
large pre-trained vision and language transformers. We propose the existence of
– “essential sparsity” defined with a sharp dropping point beyond which the
performance declines much faster w.r.t the rise of sparsity level, when we directly
remove weights with the smallest magnitudes in one-shot. We also present an
intriguing emerging phenomenon of abrupt sparsification during the pre-training
of BERT, i.e., BERT suddenly becomes heavily sparse in pre-training after certain
iterations. Moreover, our observations also indicate a counter-intuitive finding that
BERT trained with a larger amount of pre-training data tends to have a better ability
to condense knowledge in comparatively relatively fewer parameters. Lastly, we
investigate the effect of the pre-training loss on essential sparsity and discover that
self-supervised learning (SSL) objectives trigger stronger emergent sparsification
properties than supervised learning (SL). 

<img width="822" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/64eeacd0-c360-403a-92cc-7281e22fc77e">

<img width="836" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/ca3b1dbe-bc1c-45d9-a6ea-d1d0c991e997">

<img width="827" alt="image" src="https://github.com/VITA-Group/essential_sparsity/assets/6660499/a1e435e7-3082-42ec-9fc9-588a8084fa27">



