import os
import torch
import gradio as gr
import tiktoken

from model import Config, DeepSeek_V3_Model, generate_text, clean_response

# UI Metadata 
TITLE = "ChatSPE - Petroleum Engineering Assistant"
DESCRIPTION = (
    "**Note**: This model is for educational purposes only. "
    "It was pretrained on only two books and fine-tuned on just ten samples, "
    "so response quality may be **extremely limited.**"
)
EXAMPLES = [
    ["What is porosity?"],
    ["Explain water saturation."],
    ["What is permeability in petroleum engineering?"],
]

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer (tiktoken GPT-2)
tokenizer = tiktoken.get_encoding("gpt2")

#  Load Model 
config = Config()
ChatSPE = DeepSeek_V3_Model(config)
model_path = os.path.join(".", "chatspe.pt")
ChatSPE.load_state_dict(torch.load(model_path, map_location=device))
ChatSPE.to(device)
ChatSPE.eval()

# Chat Function 
def chat(user_message, history):
    try:
        prompt = f"Prompt: {user_message}\nResponse:"

        # Generate model output
        generated = generate_text(
            model=ChatSPE,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=80,
            temperature=0.7,
            top_k=10,
            eos_id=config.pad_token_id,
            device=device
        )

        # Ensure string and clean
        if isinstance(generated, bytes):
            generated = generated.decode("utf-8", errors="ignore")
        
        # Flatten any nested list or tuple from the model output
        if isinstance(generated, (list, tuple)):
            generated = " ".join(map(str, generated))
        
        reply = clean_response(str(generated))
        
        return reply
    
    except Exception as e:
        print(f"Error in chat function: {e}")
        return "Sorry, I encountered an error generating a response. Please try again."


# Gradio Chat UI - disable caching
demo = gr.ChatInterface(
    fn=chat,
    title=TITLE,
    description=DESCRIPTION,
    examples=EXAMPLES,
    cache_examples=False  # Disable caching
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)