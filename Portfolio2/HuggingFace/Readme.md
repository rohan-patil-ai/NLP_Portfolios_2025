# CodeParrot-DS: Causal Language Model from Scratch

This project demonstrates training a GPT-2 Language Model from scratch specifically for Python data science code generation. Unlike fine-tuning, this workflow starts with a randomly initialized model and trains it to understand the syntax and logic of libraries like **pandas**, **matplotlib**, and **scikit-learn**.

---

## 🚀 Overview

The project follows a rigorous NLP pipeline including custom data filtering, tokenization of code, and a specialized training loop using Hugging Face Accelerate.

### Key Features

- **Data Science Focus**: Scripts are filtered based on the presence of specific library keywords
- **Custom Tokenization**: Utilizes the code-search-net-tokenizer with parallel processing for efficiency
- **Weighted Loss**: Implements a custom loss function that prioritizes high-value tokens like `plt`, `pd`, and `fit`
- **Performance Optimization**: Features gradient accumulation and mixed-precision compatibility through the Accelerator

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python |
| **Framework** | PyTorch |
| **Libraries** | `transformers`, `datasets`, `accelerate`, `evaluate` |
| **Model** | GPT-2 (Causal LM) |

---

## 📖 Project Workflow

### 1. Data Preprocessing
The dataset is processed into fixed-length chunks of **128 tokens**. Using `return_overflowing_tokens`, we ensure that code snippets are not simply truncated but broken into meaningful sequences.

### 2. Model Configuration
The model is initialized with a custom `AutoConfig` rather than pre-trained weights:

- **Activation**: GELU
- **Context Length**: 128
- **Parameters**: ~124.2M

### 3. Custom Key-Token Weighted Loss
To ensure the model prioritizes the most important data science syntax, we implemented a **weighted Cross-Entropy loss**. This mathematically scales the loss for specific "key tokens" (like `pd` or `predict`), forcing the model to learn these critical identifiers more accurately during the early stages of training.

### 4. Training with Accelerate
While the notebook includes a high-level Trainer implementation, the primary focus is the custom Accelerate loop. This allows for:

- Manual gradient accumulation (8 steps)
- Gradient clipping for training stability
- Direct control over the optimization step and learning rate scheduler

---

## 📊 Results & Inference

The model's capability is tested via a text-generation pipeline. It can successfully take a natural language comment and generate valid Python code.

**Input:**
```python
# create scatter plot with x, y
```

**Output:**
```python
plt.scatter(x, y, color='red', marker='x')
```

---

## 💻 Usage

### Clone the Repo
```bash
git clone https://github.com/Vidyap13/codeparrot-ds.git
cd codeparrot-ds
```

### Install Requirements
```bash
pip install datasets transformers accelerate torch
```

### Inference
Run the generation cells in the notebook to interact with the trained model.

---

## 📁 Repository Structure
```
codeparrot-ds/
├── notebook.ipynb          # Main training and inference notebook
├── requirements.txt        # Python dependencies
├── data/                   # Dataset cache (generated during runtime)
└── README.md               # Project documentation
```

---

## 🔗 References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---
