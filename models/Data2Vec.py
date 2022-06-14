"""Data2Vec module based on https://github.com/arxyzan/data2vec-pytorch"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.EMA import EMA


class Data2Vec(nn.Module):
    """
    Data2Vec main module.
    """
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self,
                 encoder: nn.Module,
                 modality: str,
                 model_embed_dim: int,
                 ema_decay: float,
                 ema_end_decay: float,
                 ema_anneal_end_step: int,
                 average_top_k_layers: int,
                 normalize_targets: bool,
                 **kwargs):
        """
        :param encoder: transformer encoder
        :param modality: vision, audio or text
        :param model_embed_dim: Embedding dimension of transformer encoder
        :param ema_decay: EMA model decay
        :param ema_end_decay: EMA model end decay
        :param ema_anneal_end_step: Number of annealing steps for EMA model decay
        :param average_top_k_layers: Number of encoder layers to use for Data2Vec target
        :param normalize_targets: Specifies whether Dat2Vec targets are normalized
        :param kwargs: keyword arguments
        """
        super(Data2Vec, self).__init__()
        self.encoder = encoder
        assert modality in self.MODALITIES
        self.modality = modality
        self.embed_dim = model_embed_dim
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = ema_anneal_end_step
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.__dict__.update(kwargs)

        self.ema = EMA(self.encoder, device="cuda")  # Instantiate teacher encoder
        self.regression_head = self._build_regression_head()  # Instantiate regression head to predict target

    def _build_regression_head(self):
        if self.modality == 'text':
            embed_dim = self.embed_dim
            curr_dim = embed_dim
            projections = []
            for i in range(self.cfg.model.head_layers - 1):
                next_dim = embed_dim * 2 if i == 0 else curr_dim
                projections.append(nn.Linear(curr_dim, next_dim))
                projections.append(nn.GELU())
                curr_dim = next_dim

            projections.append(nn.Linear(curr_dim, embed_dim))
            return nn.Sequential(*projections)

        if self.modality in ['audio', 'vision']:
            return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        Function which to step the EMA encoder
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, student_input, teacher_input=None, mask=None):
        """
        Data2Vec forward method.
        :param student_input: Input for student encoder
        :param teacher_input: Input for teacher encoder
        :param mask: mask for student input if input is not already masked
        :return: Data2Vec model output x, y for student prediction and teacher target, respectively
        """

        encoder_out, student_hidden_states = self.encoder(student_input, mask=mask, output_hidden_states=True)
        if teacher_input is None:
            return encoder_out
        x = student_hidden_states[-1]
        with torch.no_grad():
            self.ema.model.eval()

            _, teacher_hidden_states = self.ema.model(teacher_input, mask=None, output_hidden_states=True)

            y = teacher_hidden_states[-self.average_top_k_layers:]
            if self.modality in ['vision', 'text']:  # Follow the same layer normalization procedure for text and vision
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            elif self.modality == 'audio':  # Use instance normalization for audio
                y = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2) for tl in y]
                y = sum(y) / len(y)
                if self.normalize_targets:
                    y = F.instance_norm(y).transpose(1, 2).transpose(1, 2)

        x = x[mask]
        y = y[mask]

        x = self.regression_head(x)

        return x, y
