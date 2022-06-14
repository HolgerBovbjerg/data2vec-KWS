"""
Masking strategy for Data2Vec Pretraining of KWT models.
"""

import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


class AudioMaskingGenerator:
    def __init__(self,
                 mask_prob: float,
                 mask_length: int,
                 attention_mask=None,
                 min_masks: int = 0):
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.attention_mask = attention_mask
        self.min_masks = min_masks

    def __call__(self, shape):
        batch_size, audio_size = shape
        mask = _compute_mask_indices((batch_size, audio_size),
                                     self.mask_prob,
                                     self.mask_length,
                                     self.attention_mask,
                                     self.min_masks)
        mask = torch.from_numpy(mask)
        return mask


def generate_masked_tensor(input_tensor, mask, fill=0):
    masked_tensor = torch.zeros(input_tensor.size(), device=input_tensor.device) + fill
    masked_tensor[mask] = input_tensor[mask]
    return masked_tensor


if __name__ == "__main__":
    audio_mask_generator = AudioMaskingGenerator(mask_prob=0.65,
                                                 mask_length=10,
                                                 attention_mask=None,
                                                 min_masks=1)
    print("Done!")
