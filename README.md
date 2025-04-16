
# 🤖 Fine-Tuning GPT-2 with LoRA for Text Generation

This project demonstrates how to fine-tune OpenAI's GPT-2 model using the [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) technique for efficient adaptation. The model is trained on the WikiText-2 dataset for causal language modeling and includes both an interactive Jupyter interface (`ipywidgets`) and a command-line-based generation loop.

---

## 🚀 Features

- 🔧 **LoRA-enabled Fine-Tuning** with the `peft` library
- 📚 Preprocessed **WikiText-2** dataset from Hugging Face
- ⚙️ Configurable training pipeline using `Trainer`
- 📊 Evaluation and generation support
- 🧠 Interactive UI for real-time text generation with:
  - Temperature
  - Top-k / Top-p sampling
  - Max length
- 💬 CLI-based generation loop
- ☁️ Optional: Save trained model to Google Drive
- 🌐 (Optional) Streamlit + ngrok for deployment

---

## 🧰 Tech Stack

- Python 3.7+
- PyTorch
- 🤗 Transformers
- 🤗 Datasets
- PEFT (LoRA)
- ipywidgets
- Streamlit + pyngrok (optional)
- Google Colab (for training)

---

## 📦 Installation

```bash
pip install transformers datasets peft torch ipywidgets streamlit pyngrok
```

---

## 📁 Dataset

- **WikiText-2 (raw)** from Hugging Face Datasets:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
  ```

---

## 🛠 Training Pipeline

- Uses `GPT2LMHeadModel` and `Trainer` API.
- LoRA applied to GPT-2’s `c_attn` and `c_proj` modules.
- Full preprocessing with token padding and truncation.
- Supports FP16 if CUDA is available.

```python
from peft import LoraConfig, get_peft_model
model = get_peft_model(model, LoraConfig(...))
```

---

## 🎛 Interactive Text Generation (UI)

An interactive widget-based interface for generating text with:

- Prompt input
- Max length slider
- Temperature, top-k, and top-p sampling
- Real-time feedback

```python
from IPython.display import display
import ipywidgets as widgets
# UI elements: prompt_input, sliders, generate_button
```

---

## 🧪 Interactive CLI Loop

Text generation via a terminal interface:

```bash
Prompt: The future of AI is
Max Length (default 50): 100
Temperature (default 0.7): 1.0
...
Generated Text:
"The future of AI is extremely bright..."
```

---

## 💾 Save Model

```python
model.save_pretrained("./gpt2_results")
tokenizer.save_pretrained("./gpt2_results")
```

Optionally, save to Google Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./gpt2_results /content/drive/MyDrive/gpt2_results_backup
```

---

## 📌 Streamlit App (Optional Deployment)

Streamlit + ngrok support available to deploy your model as a web app.

```bash
streamlit run app.py
```

You can set up an ngrok token using:

```python
from google.colab import userdata
authkey = userdata.get('ngrokauth')
```

---

## 👥 Contributors

- **Your Name** *(Team Leader or Author)*
- You can add your collaborators here

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to reuse, share, or modify!

---

## 🙌 Acknowledgments

- Hugging Face for Transformers & Datasets
- PEFT team for LoRA
- Google Colab for training environment

---

> 💡 *Ready to build and deploy your own fine-tuned GPT-2? Start generating some magic!*
```

Let me know if you’d like to add:
- A badge section (e.g., Colab, Python version, etc.)
- Example images/screenshots
- `requirements.txt` auto-generated from your environment  
- or turn this into a project card for GitHub!

I got you 👨‍💻
