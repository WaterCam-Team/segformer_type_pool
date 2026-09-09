import torch
import torch.nn as nn
import clip  # from OpenAI CLIP repo

class CustomPrompt(nn.Module):
    def __init__(self, clip_model, class_name='water', n_ctx=16, position='middle'):
        super().__init__()
        self.clip_model = clip_model
        self.class_name = class_name
        self.position = position
        self.n_ctx = n_ctx
        self.tokenizer = clip.tokenize
        self.ctx_dim = clip_model.token_embedding.embedding_dim
        self.dtype = clip_model.dtype

        # Learnable context tokens
        ctx_vectors = torch.randn(n_ctx, self.ctx_dim)
        #nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        def print_grad_hook(grad):
            print("[HOOK] ctx.grad mean:", grad.mean().item(), "std:", grad.std().item())

        self.ctx.register_hook(print_grad_hook)
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = prompt_prefix + " " + class_name + "."
        # Tokenize class name to extract full token embeddings including CLS and EOS
        with torch.no_grad():
            self.class_token_ids = self.tokenizer(prompts).to(clip_model.token_embedding.weight.device)
            self.class_embed_full = clip_model.token_embedding(self.class_token_ids).squeeze(0).type(self.dtype)

        self.token_prefix = self.class_embed_full[:1]      # SOS
        self.token_suffix = self.class_embed_full[1 + n_ctx:]      # CLS, EOS
        self.name_len = 1
        # self.class_embed = self.class_embed_full[1:-1]   # Class word(s)

    def forward(self):
        ctx = self.ctx  # [n_ctx, dim]
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.position == 'middle':
            half = self.n_ctx // 2
            class_embed = suffix[:self.name_len]
            suffix = suffix[self.name_len:]
            ctx_i_half1 = ctx[:half, :]
            ctx_i_half2 = ctx[half:, :]
            prompt = torch.cat(
                [
                    prefix,     # (1, 1, dim)
                    ctx_i_half1,  # (1, n_ctx//2, dim)
                    class_embed,      # (1, name_len, dim)
                    ctx_i_half2,  # (1, n_ctx//2, dim)
                    suffix,     # (1, *, dim)
                ],
                dim=0)
        elif self.position == 'end':
            prompt = torch.cat([
                prefix,
                ctx,
                suffix,
            ], dim=0)
        else:  # 'front'
            class_embed = suffix[:self.name_len]
            suffix = suffix[self.name_len:]
            prompt = torch.cat(
                [
                    prefix,  # (1, 1, dim)
                    class_embed,   # (1, name_len, dim)
                    ctx,     # (1, n_ctx, dim)
                    suffix,  # (1, *, dim)
                ],
                dim=0)
        prompt = prompt.unsqueeze(0)
        text_feat = prompt + self.clip_model.positional_embedding.type(self.dtype)
        text_feat = text_feat.permute(1, 0, 2).type(self.dtype)  # NLD -> LND
        text_feat = self.clip_model.transformer(text_feat)
        text_feat = text_feat.permute(1, 0, 2)  # LND -> NLD
        text_feat = self.clip_model.ln_final(text_feat).type(self.dtype)
        text_feat = text_feat[torch.arange(text_feat.shape[0]), self.class_token_ids.argmax(dim=-1)] @ self.clip_model.text_projection
        
        return text_feat  # [sequence_len, dim]


if __name__ == '__main__':
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    coop_prompt = CustomPrompt(clip_model, class_name="water", n_ctx=16, position="end").to(device)

    # Forward pass to get prompt embedding
    text_feat = coop_prompt()  # [sequence_len, dim]
    print("Prompt shape:", text_feat.shape)
