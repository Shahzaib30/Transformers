from xml.parsers.expat import model

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, n_dimension, n_head):
        super().__init__()

        self.multi_attention = nn.MultiheadAttention(n_dimension, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(n_dimension)
        self.ffn = nn.Sequential(
            nn.Linear(n_dimension, n_dimension * 4),
            nn.ReLU(),
            nn.Linear(n_dimension * 4, n_dimension)
        )
        self.norm2 = nn.LayerNorm(n_dimension)

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        atten_out, _ = self.multi_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + atten_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x 
    
class TinyStories(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_head, n_layer):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Embedding(512, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.head = nn.Linear(n_embedding, vocab_size)

    def forward(self, x, target=None):
        batch_size, seq_len = x.size()
        token_emb = self.token_embedding(x)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_emb = self.position_embedding(position_ids)

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if target is not None:
            B , T, C = logits.size()
            logits = logits.view(B * T, C)  
            target = target.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, target)

        return logits, loss
    

def get_batch(data_iter, batch_size, block_size):
    input_idx = []
    target_idx = []
    for _ in range(batch_size):
        story = next(data_iter)['text']
        tokens = tokenizer.encode(story, truncation=True, max_length=block_size)
        if len(tokens) < block_size + 1:
            tokens += [tokenizer.eos_token_id] * (block_size + 1 - len(tokens))

        else:
            tokens = tokens[:block_size + 1]

        input_idx.append(tokens[:-1])
        target_idx.append(tokens[1:])

    return torch.tensor(input_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)





if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming = True)
    data_iter = iter(dataset)
    sample_story = next(data_iter)['text']


    n_embedding = 128
    n_head = 8
    n_layer = 6
    block_size = 256
    vocab_size = tokenizer.vocab_size
    model = TinyStories(vocab_size, n_embedding, n_head, n_layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Training Started ....")
    model.train()

    for step in range(10000):
        xb, yb = get_batch(data_iter, batch_size= 16, block_size=block_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), "tiny_gpt2.pth")

    torch.save(model.state_dict(), "tiny_gpt2.pth")
    print("Model saved as tiny_gpt2.pth")
        

