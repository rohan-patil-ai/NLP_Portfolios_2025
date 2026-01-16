Team: Harsha Satish, Riya Biju, Rohan Sanjay Patil, Vidya Padmanabha

This folder collects three NLP/ML projects that explore generation, classical ML (TF-IDF) and model training on the Hugging Face stack.

## Contents

- Poem_Generator/
	- `app.py` — small app to generate poems/songs using a large language model.
	- `song_writer.py` — helper utilities and prompt templates.
	- Readme.md — usage notes for the poem generator.

- TFIDF/
	- `Tf_idf_Movie.ipynb` — notebook implementing a movie genre classifier using TF-IDF vectorization and inspecting correlated unigrams/bigrams.

- HuggingFace/
	- `Training_a_causal_language_model_from_scratch_(PyTorch).ipynb` — notebook showing how to train a causal language model from scratch using the Hugging Face ecosystem (datasets, transformers, accelerate).

## Poem_Generator (song writer)

Summary
- We built a small poem/song generator that uses a large LLM (llama3:8b in the project experiments) and a structured prompt to produce creative, multi-part lyrics.

Prompt (used in the project)
```
prompt = f"""
		You are a professional and creative songwriter.
		Your task is to write a complete song with a clear structure.
    
		The song must be inspired by these three words:
		1. {word1}
		2. {word2}
		3. {word3}
    
		Please structure the song with:
		- Verse 1
		- Chorus
		- Verse 2
		- Chorus
		- Bridge
		- Chorus
    
		Generate the lyrics now.

To give a song/poem
"""
```

Notes
- The experiments used the `llama3:8b` family as the generative model. Replace with your preferred LLM or hosted API key if you want to run locally or on a hosted service.

How to run
1. Create a Python virtual environment and install needed packages (example):

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if a requirements file exists; otherwise install your LLM client SDK
```

2. Edit `app.py` to configure your model endpoint or local model weights and any API keys.

3. Run the app:

```zsh
python Poem_Generator/app.py
```

Replace model name and client configuration as needed (for example, a hosted LLM provider or local llama runtime).

## TFIDF: Movie genre classifier

Summary
- Implemented a movie-genre classifier using TF-IDF vectorization over movie text (title/description) and a classical classifier.
- The notebook also inspects correlated unigrams and bigrams to show which tokens/features are most predictive of genres.

Files
- `TFIDF/Tf_idf_Movie.ipynb` — run this notebook to reproduce experiments, view plots, and inspect feature correlations.

How to run
1. Open the notebook in Jupyter or VS Code:

```zsh
jupyter notebook TFIDF/Tf_idf_Movie.ipynb
# or
code TFIDF/Tf_idf_Movie.ipynb
```

2. Install common data-science dependencies if you don't have them:

```zsh
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

3. Run cells top-to-bottom. The notebook includes plots and summary tables for correlated unigrams/bigrams.

## Hugging Face: Training a causal language model from scratch

Summary
- The notebook `HuggingFace/Training_a_causal_language_model_from_scratch_(PyTorch).ipynb` walks through:
	- loading datasets (streaming and filtering),
	- tokenization and chunking for causal LM training,
	- building a GPT-2 style config and model from scratch,
	- data collator setup, and
	- training with `accelerate` for mixed-device support.

Key notebooks/steps
- Install required libraries (notebook shows these commands):

```python
!pip install datasets evaluate transformers[sentencepiece]
!pip install accelerate
# optional: git-lfs if saving large model files
!apt install git-lfs
```

- Follow the notebook to set up `tokenizer`, `GPT2LMHeadModel` config, data collator, and the training loop. The notebook also demonstrates saving/pushing to the Hugging Face Hub.

How to run
1. Open the notebook in Jupyter or Colab (Colab is recommended for GPU access):

```zsh
jupyter notebook HuggingFace/Training_a_causal_language_model_from_scratch_(PyTorch).ipynb
```

2. If running locally, ensure you have a GPU and the correct PyTorch + CUDA build. Use the notebook's first cells to install/validate the environment.

3. The notebook includes `notebook_login()` calls — log in to push checkpoints or models to the Hugging Face Hub.

Notes and caveats
- Training a causal model from scratch is compute-intensive. The notebook is tuned for smaller proof-of-concept runs (reduced dataset sizes, short epochs) — change batch sizes and epochs carefully.
- When pushing to the Hub, ensure you have configured `git-lfs` and have appropriate token/permissions.

## Reproducibility & Dependencies

- Each subfolder includes the code/notebook needed to reproduce the experiments. For best results, create a fresh virtual environment and install packages listed in each notebook or add a `requirements.txt` per project.
- Typical packages used across the portfolio:
	- datasets, transformers, accelerate, evaluate
	- numpy, pandas, scikit-learn, matplotlib, seaborn

## Contact / Credits

Team: Harsha Satish, Riya Biju, Rohan Sanjay Patil, Vidya Padmanabha

If you need help reproducing any experiment or want a README expanded with explicit environment manifests, let us know and we can add per-project requirements files and run scripts.
