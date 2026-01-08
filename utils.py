import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Đã cố định Seed: {seed}")


def decode_predictions(preds, idx2char, beam_width=1):
    """
    Decode CTC predictions to text strings.
    
    Args:
        preds: Tensor of shape [B, T, C] containing log-probs/probs OR [B, T] for greedy
        idx2char: Dictionary mapping indices to characters
        beam_width: If > 1, use beam search
    
    Returns:
        List of decoded strings
    """
    if beam_width <= 1:
        # Greedy Decoding
        if len(preds.shape) == 3:
            preds = torch.argmax(preds, dim=2)
        
        result_list = []
        for p in preds:
            pred_str = ""
            last_char = 0
            for char_idx in p:
                c = char_idx.item()
                if c != 0 and c != last_char:
                    pred_str += idx2char[c]
                last_char = c
            result_list.append(pred_str)
        return result_list
    else:
        # Simplified CTC Beam Search
        return ctc_beam_search_decode(preds, idx2char, beam_width)


def ctc_beam_search_decode(log_probs, idx2char, beam_width=10):
    """
    Standard CTC Beam Search decoding with Top-K optimization.
    """
    batch_size, seq_len, num_classes = log_probs.size()
    results = []

    # Pre-select top candidates per step to speed up inner loops
    # Taking top min(beam_width * 2, num_classes) candidates
    topk_k = min(beam_width * 2, num_classes)
    topk_log_probs, topk_indices = log_probs.topk(topk_k, dim=2)

    for b in range(batch_size):
        # beams: {prefix: (p_blank, p_nonblank)}
        beams = {(): (0.0, -float('inf'))} 
        
        for t in range(seq_len):
            new_beams = {}
            
            # Use pre-computed top-k candidates for this batch and time step
            step_topk_log_probs = topk_log_probs[b, t]
            step_topk_indices = topk_indices[b, t]
            
            for prefix, (p_b, p_nb) in beams.items():
                for k in range(topk_k):
                    char_idx = step_topk_indices[k].item()
                    char_log_prob = step_topk_log_probs[k].item()
                    
                    if char_idx == 0: # Blank
                        curr_p_b, curr_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                        p_total = np.logaddexp(p_b, p_nb)
                        new_beams[prefix] = (np.logaddexp(curr_p_b, p_total + char_log_prob), curr_p_nb)
                    else:
                        new_prefix = prefix + (char_idx,)
                        if len(prefix) > 0 and char_idx == prefix[-1]:
                            # Repeat character
                            curr_p_b, curr_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                            new_beams[prefix] = (curr_p_b, np.logaddexp(curr_p_nb, p_nb + char_log_prob))
                            
                            n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                            new_beams[new_prefix] = (n_p_b, np.logaddexp(n_p_nb, p_b + char_log_prob))
                        else:
                            # Different character or first character
                            n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                            p_total = np.logaddexp(p_b, p_nb)
                            new_beams[new_prefix] = (n_p_b, np.logaddexp(n_p_nb, p_total + char_log_prob))
            
            # Prune to beam width
            beams = dict(sorted(new_beams.items(), 
                               key=lambda x: np.logaddexp(x[1][0], x[1][1]), 
                               reverse=True)[:beam_width])
            
        # Select best prefix
        best_prefix = max(beams.keys(), key=lambda x: np.logaddexp(beams[x][0], beams[x][1]))
        results.append("".join([idx2char[c] for c in best_prefix]))
        
    return results
