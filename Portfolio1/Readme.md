# Portfolio 1: Research Paper Analysis

## 📄 Featured Paper
**Title:** [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)  
**Authors:** Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin (2003)  
**Core Theme:** Overcoming the Curse of Dimensionality through Distributed Representations.

---

## 🎯 Project Objective
This portfolio focuses on the study and presentation of the foundational paper that introduced **Word Embeddings** to the world of NLP. Our goal was to understand how moving from discrete n-grams to continuous vector spaces revolutionized language modeling.

## 🧠 Key Insights from the Study

### 1. The Problem: Curse of Dimensionality
Traditional n-gram models struggle because:
* They require exact phrase matches to make predictions.
* With a 100,000-word vocabulary, a 10-word sequence creates $10^{50}$ possible combinations—most of which are never seen during training.
* They don't understand that "dog" and "cat" are semantically similar.

### 2. The Solution: Distributed Representations
Bengio et al. proposed learning a **distributed representation** for words:
* **Feature Vectors:** Each word is mapped to a vector in $\mathbb{R}^m$.
* **Semantic Similarity:** Similar words (cat ≈ dog) result in similar feature vectors.
* **Generalization:** By learning these vectors, the model automatically understands sentences it has never seen (e.g., if it trains on *"The cat is walking"*, it can predict *"A dog is running"*).

### 3. Model Architecture
The NPLM architecture consists of:
* **Input Layer:** Mapping word indices to feature vectors (Matrix $C$).
* **Hidden Layer:** A `tanh` activation layer to capture non-linear relationships.
* **Output Layer:** A `Softmax` function that produces a probability distribution over the entire vocabulary.

## 📊 Results Summary
The Neural Network approach significantly outperformed traditional back-off n-gram models:
* **Brown Corpus:** Reduced test perplexity by **24%** (252 vs 336).
* **AP News:** Reduced test perplexity by **8%** (109 vs 117).

---

## 📂 Included Files
* [Research Paper_NPLM.pdf](./Research%20Paper_NPLM.pdf) - The original 2003 JMLR paper.
* [NLPM PPT.pptx](./NLPM%20PPT.pptx) - Our team's presentation deck breaking down the concepts, math, and results.

## Contact / Credits

Team: Harsha Satish, Riya Biju, Rohan Sanjay Patil, Vidya Padmanabha
