import torch
import torch.nn as nn
from transformers import AutoTokenizer

from GPT2 import TinyStories
from GPT2 import TinyStories 


n_embedding = 128
n_head = 8
n_layer = 6
block_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

model = TinyStories(vocab_size, n_embedding, n_head, n_layer).to(device)
model.load_state_dict(torch.load("tiny_gpt2.pth"))
model.eval() 

# --- THE GENERATOR ---
def generate_story(prompt, max_new_tokens=1000):
    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] 
        logits = logits / 0.7
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, next_token), dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        
    return tokenizer.decode(idx[0], skip_special_tokens=True)

# --- TEST IT ---
my_prompt = "One day, a small bird"
print(f"\nPROMPT: {my_prompt}")
print(f"STORY: {generate_story(my_prompt)}")