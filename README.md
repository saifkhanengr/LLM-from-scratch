In this repo, we will go through all the math and code from scratch of a Large Language Model (specifically DeepSeek V3).

* The `Notebook.ipynb` file walks you through everything step by step from scratch.
* The utils.py file contains the same essential code as the notebook, but organized in one place for convenience.

Below is what will be covered in the `Notebook.ipynb` (with math and code):

1. **Text to Token IDs:**
   *  Example: "I love AI!" → [1, 2, 3, 0]

2. **Token IDs to Token Embeddings:**
   *  Token IDs are just random numbers; they don't capture meaning.
   *  Embeddings turn Token IDs into meaningful vectors, where similar words get nearby values, e.g., man ≈ 1.0, king ≈ 1.1 (same gender), while woman ≈ -1.0, queen ≈ -1.1 (opposite of male gender). In short, embeddings represent the meaning of text.

3. **Token Embeddings to Q, K, V:**

    *  Embeddings capture the meanings of words, but the model needs to determine how each word relates to other words in the sentence.
    *  This is done by projecting (linear transformation of) embeddings into Query (Q), Key (K), and Value (V) vectors.

    *  For example, in the sentence: *Sarah visits the bank to deposit money while Zarah visits the bank to sit by the water.*

        *  **Query (Q):** Represents a specific word (e.g., bank) and asks how it relates to other words in the sentence. Is it a financial institution or a river?

        *  **Key (K):** Represents all words in the sentence and is used to measure the relevance (attention score) of each word to the Query. So, "bank" (financial) will get a high attention score when Q is "Sarah".

        *  **Value (V):** Contains the actual information of each word and provides the final contextual meaning. So, the word "bank" near "Sarah" means a financial institute rather than a river bank.


4. **Positional Encodings (RoPE):**
   *  The embeddings don't know where in the sequence the token (word) appears
   *  Without position info, **The cat chased the dog** would look the same as **The dog chased the cat.**
   * So we inject positional information. DeepSeek used Rotary Position Embeddings (RoPE) only for Q and K.

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
