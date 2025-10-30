import torch
from torch.nn import functional as F


def get_most_likely_row(tokens, mask, logits):
    """
    Helper function for HellaSwag eval
    Takes tokens, mask, and logits, returns the index of the completion with the lowest loss

    Args:
        tokens: input tokens
        mask: mask indicating completion region
        logits: model logits

    Returns:
        index of the most likely completion
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def generate_samples(model, tokenizer, prompt, num_return_sequences=4, max_length=32, device='cuda'):
    """
    Generate text samples from the model

    Args:
        model: the language model
        tokenizer: tokenizer (e.g., tiktoken encoder)
        prompt: text prompt to start generation
        num_return_sequences: number of samples to generate
        max_length: maximum length of generated sequence
        device: device to run generation on

    Returns:
        list of generated text samples
    """
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits, loss = model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # decode the generated sequences
    samples = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        samples.append(decoded)

    return samples
