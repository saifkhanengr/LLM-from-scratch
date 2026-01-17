In this repo, we will go through all the math and code from scratch of a Large Language Model (specifically DeepSeek V3).

* The `Notebook.ipynb` file walks you through everything step by step from scratch.
* The `utils.py` file contains the same essential code as the notebook, but organized in one place for convenience.
* The `ChatSPE` folder contains three files: `app.py`, `model.py`, and `requirements.txt` (and forth file would be model), all are used for production. For all the glory details, see Section 13 in the `Notebook.ipynb`.

Below is what is covered in the `Notebook.ipynb` (with math and code):

1. **Text to Token IDs:**
2. **Token IDs to Token Embeddings:**
3. **Token Embeddings to Q, K, V:**
4. **Positional Encodings (RoPE):**
5. **Multi-Head Latent Attention**
6. **Mixture of Expert (MoE)**
7. **Multi-Token Prediction (MTP)**
8.  
    &nbsp;&nbsp;a. **DeepSeek V3 Block (Single-Block Transformer)**  
    &nbsp;&nbsp;b. **DeepSeek V3 Encoder (Multi-Block Transformer)**  
    &nbsp;&nbsp;c. **DeepSeek V3 Model (Full Model)**
9. **Pre-Training the Model**
10. **Autoregressive Text Generation**
11. **Fine-Tuning the Model**
12. **Chat with your Model (Q&A)**
13. **Deploying Model to Production**
