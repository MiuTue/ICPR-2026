import os
import random
import numpy as np
import torch
import re


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


def brazil_plate_score(plate):
    """
    Calculate a validity score based on Brazilian/Mercosur license plate format.
    Format: 3 uppercase letters + 4 digits (e.g. "AVL5215")

    Higher score = more likely valid.
    """
    if not plate:
        return -10.0

    # Exact match: 3 letters + 4 digits
    if re.match(r'^[A-Z]{3}\d{4}$', plate):
        return 2.0

    # Minor variation: swapped characters (common OCR error)
    # e.g. "JRN" vs "JHN" — both get partial credit
    if re.match(r'^[A-Z]{3}\d{4}$', plate) is not None or \
       re.match(r'^[A-Z0-9]{7}$', plate):
        return 0.5

    # Penalty for wrong length
    if len(plate) < 7 or len(plate) > 7:
        return -2.0

    # Neutral for other alphanumeric strings
    if re.match(r'^[A-Z0-9]+$', plate):
        return 0.0

    return -5.0


def decode_predictions(preds, idx2char, beam_width=1, use_format_filter=False):
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
                    pred_str += idx2char.get(c, '')
                last_char = c
            result_list.append(pred_str)
        return result_list
    else:
        # Beam Search with optional Format Filter
        return _ctc_beam_search_decode(preds, idx2char, beam_width, use_format_filter)


def _ctc_beam_search_decode(log_probs, idx2char, beam_width=1, use_format_filter=False):
    """
    CTC Beam Search decoding with Top-K optimization and Brazilian plate filter.
    """
    batch_size, seq_len, num_classes = log_probs.size()
    results = []

    topk_k = min(beam_width * 2, num_classes)

    for b in range(batch_size):
        log_probs_b = log_probs[b]
        if torch.isnan(log_probs_b).any():
            log_probs_b = torch.nan_to_num(log_probs_b, nan=-10.0, posinf=0.0, neginf=-10.0)

        beams = {(): (0.0, -float('inf'))}

        for t in range(seq_len):
            step_log_probs = log_probs_b[t]
            topk_log_probs_step, topk_indices_step = step_log_probs.topk(topk_k)

            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                for k in range(topk_k):
                    char_idx = topk_indices_step[k].item()
                    char_log_prob = topk_log_probs_step[k].item()

                    if char_idx == 0:  # Blank
                        curr_p_b, curr_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                        p_total = np.logaddexp(p_b, p_nb)
                        new_beams[prefix] = (
                            np.logaddexp(curr_p_b, p_total + char_log_prob),
                            curr_p_nb
                        )
                    else:
                        new_prefix = prefix + (char_idx,)
                        if len(prefix) > 0 and char_idx == prefix[-1]:
                            # Same as last: extend blank or new char
                            curr_p_b, curr_p_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                            new_beams[prefix] = (
                                curr_p_b,
                                np.logaddexp(curr_p_nb, p_nb + char_log_prob)
                            )
                            n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                            new_beams[new_prefix] = (
                                n_p_b,
                                np.logaddexp(n_p_nb, p_b + char_log_prob)
                            )
                        else:
                            n_p_b, n_p_nb = new_beams.get(new_prefix, (-float('inf'), -float('inf')))
                            p_total = np.logaddexp(p_b, p_nb)
                            new_beams[new_prefix] = (
                                n_p_b,
                                np.logaddexp(n_p_nb, p_total + char_log_prob)
                            )

            beams = dict(sorted(
                new_beams.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:beam_width])

        # Select best prefix
        candidates = []
        for prefix, (p_b, p_nb) in beams.items():
            score = np.logaddexp(p_b, p_nb)
            text = "".join([idx2char.get(c, '') for c in prefix])
            candidates.append((text, score))

        # Apply format filter if enabled
        if use_format_filter:
            filtered = [(t, s) for t, s in candidates if brazil_plate_score(t) >= 0.0]
            if filtered:
                best = max(filtered, key=lambda x: x[1] + brazil_plate_score(x[0]))
            else:
                best = max(candidates, key=lambda x: x[1])
        else:
            best = max(candidates, key=lambda x: x[1])

        results.append(best[0])

    return results
