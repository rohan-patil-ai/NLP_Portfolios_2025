# Teaching a 2B Model to Think 🧠

> **71.3% accuracy on GSM8K using a 2-billion parameter model trained with GRPO on a single TPU**

[![Kaggle](https://img.shields.io/badge/Kaggle-Model-20BEFF?logo=kaggle)](https://www.kaggle.com/models/rohanpatil27/gemma-2b-grpo-multisession-final)
[![Tunix](https://img.shields.io/badge/Built%20with-Tunix-4285F4)](https://www.kaggle.com/competitions/google-tunix-hackathon)

**Google Tunix Hackathon Submission**

---

## 🎯 What We Built

This project demonstrates how to train a lightweight 2B parameter model (Gemma 2B) to solve grade-school math problems through structured reasoning. Using Google's Tunix library and GRPO (Group Relative Policy Optimization), we achieved **71.3% accuracy** on the GSM8K dataset—approaching the performance of models 5-10x larger.

**Key Innovation:** Combining efficient RL training (GRPO) with inference-time consensus voting to maximize small model capability under hardware constraints.

---

## 📊 Results at a Glance

| Metric | Value |
|--------|-------|
| **Final Accuracy** | 71.3% |
| **Baseline (Single-Pass)** | 63.0% |
| **Consensus Improvement** | +8.3% |
| **Model Size** | 2B parameters |
| **Hardware** | Single TPU v5e-8 |
| **Training Sessions** | 3 (relay training) |

### Performance Across Three Runs

| Run | Baseline | Consensus | Improvement | Notes |
|-----|----------|-----------|-------------|-------|
| 1 | 58.0% | 69.9% | +11.9% | Most stable learning |
| 2 | 54.0% | 62.8% | +8.8% | Higher initial format errors |
| **3** | **63.0%** | **71.3%** | **+8.3%** | **Best absolute performance** |

---

## 🏗️ Architecture

### Why GRPO?

Traditional RLHF requires:
- **Actor model**: 2B parameters
- **Critic model**: 2B parameters
- **Total**: 4B+ parameters → Won't fit on single TPU ❌

GRPO requires:
- **Actor model**: 2B parameters
- **Self-comparison**: No additional model needed
- **Total**: 2B parameters → Fits perfectly ✅

### Training Pipeline

```
┌─────────────────────────────────────────────────┐
│  Session 1: Train 500 samples → Checkpoint      │
│  Session 2: Load + Train 500 → Checkpoint       │
│  Session 3: Load + Train 500 → Final Model      │
└─────────────────────────────────────────────────┘
                     ↓
        Active Review (1:1 new:old mix)
                     ↓
          Prevents Catastrophic Forgetting
```

---

## 🎓 Key Techniques

### 1. **Composite Reward Function**

We designed a reward system that emphasizes both correctness and process:

```python
def compute_reward(output, ground_truth):
    reward = 0
    
    # Correctness (main signal)
    if extract_answer(output) == ground_truth:
        reward += 25
    
    # Structured reasoning
    if has_reasoning_tags(output):
        reward += 2
    if has_answer_tags(output):
        reward += 2
    
    # Shows work
    if has_mathematical_operators(output):
        reward += 1
    
    # Penalties
    if has_format_violations(output):
        reward -= 15
    
    return reward
```

**Critical hyperparameter:** `beta = 0.04` (KL penalty)
- Too high (0.1) → Model too explorative
- Too low (0.0) → Model too conservative
- Sweet spot (0.04) → Balanced exploration

### 2. **Active Review (Anti-Forgetting Strategy)**

When training across multiple sessions, the model would forget previously learned concepts. Our solution:

```python
# For every batch of new samples
new_samples = dataset[current_index:current_index+500]
old_samples = random.sample(all_previous_samples, 500)

training_batch = new_samples + old_samples  # 1:1 ratio
```

This simple technique maintained knowledge retention without exploding dataset size.

### 3. **Consensus Voting at Inference**

The model's errors were often stochastic (random arithmetic mistakes), but correct reasoning was consistent. We exploited this by generating multiple completions:

```python
temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
completions = [generate(prompt, temp=t) for t in temperatures]
answers = [extract_answer(c) for c in completions]
final_answer = majority_vote(answers)
```

**Results:**
- Fixed: 21 errors
- Broke: 14 correct answers  
- Net gain: +7 problems (+8.3% accuracy)

---

## 📈 What We Learned

### Pattern 1: Optimal Reasoning Length Emerged

Without explicit training, the model learned that **50-60 words** was the sweet spot for reasoning:

- **Too short** (< 30 words): Insufficient reasoning
- **Too long** (> 80 words): Model gets lost in its own logic
- **Just right** (50-60 words): Highest accuracy

### Pattern 2: Consensus Works Best as "Polish"

Consensus voting is most effective when the baseline model is already capable (55-65% range). It's a reliability enhancement, not a rescue mechanism.

| Baseline Quality | Consensus Effectiveness |
|-----------------|------------------------|
| Weak (< 50%) | Limited help, high "broke" rate |
| Mid-tier (55-65%) | **Optimal gains** |
| Strong (> 65%) | Diminishing returns |

### Pattern 3: Format vs. Logic Trade-offs

Different training runs emphasized different aspects:

- **Run 1**: Best format compliance (84%/83%) but 25% wrong answers
- **Run 2**: Higher format errors but improved logic over time
- **Run 3**: Balanced both, achieving best overall performance

This suggests training dynamics aren't uniform—some runs "learn" structure faster, others learn reasoning faster.

### Pattern 4: The Capability Ceiling

Even in our best run, **11 problems** remained unsolvable (both baseline and consensus failed). These represent genuine gaps in the model's reasoning capability—concepts we didn't successfully train for.

**Example hard cases:**
- Multi-step problems with unit conversions
- Problems requiring external world knowledge
- Complex word problems with implicit assumptions

---

## 🔍 Critical Assessment

### What Works Well ✅

1. **GRPO scales RL to resource-constrained environments** - Full RL training on single TPU
2. **Active Review prevents catastrophic forgetting** - Essential for multi-session training
3. **Consensus voting adds reliability** - 8-12% improvement with 5x inference cost
4. **Small models can compete** - 71.3% approaches much larger model performance

### Limitations & Trade-offs ⚠️

1. **Still can't solve 20-30% of problems** - Hard capability limits exist
2. **Consensus adds 5x inference cost** - Trade-off between accuracy and speed
3. **Format vs logic learning unclear** - Don't fully understand training dynamics

### Honest Take

This is a **practical system with clear trade-offs**. The real contribution is demonstrating that inference-time compute can compensate for model size in structured reasoning tasks.

---

## 🚀 Getting Started

### Prerequisites

```bash
# TPU environment (Kaggle or Google Cloud)
pip install tunix jax flax optax
```

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/gemma-2b-reasoning.git
cd gemma-2b-reasoning
```

2. **Download the dataset:**
```python
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
```

3. **Run training (Session 1):**
```bash
python conquering-gemma-session-1.ipynb
# This will train for ~2 hours and save checkpoint
```

4. **Continue training (Sessions 2-3):**
```bash
python conquering-gemma-session-2.ipynb
python conquering-gemma-session-3.ipynb
```

5. **Evaluate with consensus voting:**
```bash
python evaluate_consensus.py --model_path checkpoints/session_3_final
```

### Using Pre-trained Models

Download our trained checkpoints from Kaggle:

```python
from kaggle_hub import model_download

# Foundation model (after Session 1)
foundation_path = model_download("rohanpatil27/gemma-2b-grpo-multisession-foundation")

# Final model (after Session 3)
final_path = model_download("rohanpatil27/gemma-2b-grpo-multisession-final")
```

---

## 📁 Repository Structure

```
├── conquering-gemma-session-1.ipynb  # Initial training session
├── conquering-gemma-session-2.ipynb  # Continued training with Active Review
├── conquering-gemma-session-3.ipynb  # Final training session
├── Results/                           # Training logs and visualizations
│   ├── evaluation_dashboard_1.png
│   ├── evaluation_dashboard_2.png
│   └── evaluation_dashboard_3.png
├── README.md                          # This file
```

---

## 🎯 Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `beta` (KL penalty) | 0.04 | Balance exploration vs stability |
| `learning_rate` | 1e-5 | Standard for GRPO |
| `batch_size` | 16 | Constrained by TPU memory |
| `active_review_ratio` | 1:1 | New samples : Old samples |
| `consensus_temperatures` | [0.2, 0.4, 0.6, 0.8, 1.0] | Diversity in sampling |
| `num_epochs_per_session` | ~5000 steps | Based on 500 samples |

---

## 🤝 Contributing

This is a hackathon submission project, but suggestions and improvements are welcome! Feel free to:

- Open issues for bugs or questions
- Submit PRs for improvements
- Share your own training runs and results
- Experiment with different hyperparameters

---

## 📖 Additional Resources

- **Tunix Documentation**: [Google DeepMind Tunix](https://github.com/google-deepmind/tunix)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **GSM8K Dataset**: [Grade School Math 8K](https://github.com/openai/grade-school-math)
- **Gemma Models**: [Google Gemma](https://ai.google.dev/gemma)

---

## 📞 Contact

- **Email**: rohansanjay.patil@study.thws.de
- **Email**: vidya.padmanabha@study.thws.de
- **Email**: harsha.sathish@study.thws.de
- **Email**: riya.biju@study.thws.de

**Built for Google Tunix Hackathon 2026**
