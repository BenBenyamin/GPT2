
# GPT-2 From Scratch

This repository offers a PyTorch-based implementation of the GPT-2 language model, built entirely from scratch.
It includes custom modules for attention mechanisms, transformer blocks, loss computation, optimization, and training routines. The model was trained on 40B tokens from the [FineWebEdu-10B Dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

---

## Summary

The project was based on the GPT-2 and GPT-3 papers, from which the training methodology—including the cosine learning rate scheduler and key hyperparameters—was adopted. The implemented model follows the GPT-2 Small configuration, as seen below.

| Hyperparameter       | Value             |
|----------------------|-------------------|
| Model Architecture   | GPT-2 Small       |
| Number of Layers     | 12                |
| Hidden Size          | 768               |
| Attention Heads      | 12                |
| Sequence Length      | 1024              |
| Batch Size           | 524288            |
| Total Training Time  | ~80 hours         |
| GPU                  | 2 x NVIDIA A6000  |
| Best Validation Loss | 2.99              |

Example text generation:

```txt

Hello, I'm a language model, I'm an author, I'm just starting a new language, I'm a computer scientist, I've been doing things over the past few years, now I'm beginning a new area of research, I'm doing research, I'm taking a field trip. My goal is to discover the answer to the questions in the context of what happens when you write computers. So that I understand what the problem is has done. Now it's so different from doing the computer science. So the language, we use, is to think, think how?
A computer scientist is someone whose job is to design, model, and implement computer systems. They are also known for their creativity, creativity, critical thinking capabilities. They are excellent in creative problem solving, they are able to see things before they can be solved, their ability to think in a natural, natural way, and their ability to make changes to the world, and their ability to have a lot of imagination
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (highly recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BenBenyamin/GPT2.git
   cd GPT2
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```bash
GPT2/
├── dataset.py                # Dataset loading and preprocessing
├── extra/                    # Supplementary scripts and experiment logs
│   ├── load_gpt2_weights.py  # Load Hugging Face weights into the custom model
│   ├── tokenize_dataset.py   # Tokenization pipeline for raw dataset
│   └── tensorboard/
│       └── runs              # TensorBoard run
├── generation_log.txt        # Sample generated outputs across training
├── loss.py                   # Custom GPT-2 loss function
├── model.py                  # Model architecture definitions
├── optimizer.py              # Cosine scheduler and AdamW setup
├── README.md                 # Project documentation
├── requirements.txt          # Python package dependencies
├── resume.py                 # Checkpoint loading
├── train.py                  # Training loop and validation evaluation
└── utils.py                  # General-purpose helper functions

```

---


**Training:**

```bash
python train.py
```
---

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)  
- [FineWebEdu-10B Dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)  
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)  
- [Attention is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)  
- [Andrej Karpathy’s GPT Tutorial (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)  


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
