import copy

import torch
import torch.nn as nn


class EMA:
    """
    Modified version of class fairseq.models.ema.EMA.
    """

    def __init__(self, model: nn.Module, device=None, skip_keys=None, ema_decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        if device is not None:
            self.model.to(device)
        self.device = device
        self.skip_keys = skip_keys or set()
        self.decay = ema_decay
        self.num_updates = 0

    def step(self, new_model: nn.Module):
        ema_state_dict = {}
        ema_params = self.model.state_dict()
        for key, param in new_model.state_dict().items():
            ema_param = ema_params[key].float()
            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.decay)
            ema_state_dict[key] = ema_param
        self.model.load_state_dict(ema_state_dict, strict=False)
        self.num_updates += 1

    def restore(self, model: nn.Module):
        d = self.model.state_dict()
        model.load_state_dict(d, strict=False)
        return model

    def _set_decay(self, decay):
        self.decay = decay

    def get_decay(self):
        return self.decay

    @staticmethod
    def get_annealed_rate(start, end, curr_step, total_steps):
        r = end - start
        pct_remaining = 1 - curr_step / total_steps
        return end - r * pct_remaining


if __name__ == "__main__":
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(10),
                nn.Linear(10, 10),
                nn.Linear(10, 2)
            )

        def forward(self, x):
            return self.net(x)


    import torch.optim as optim

    model = Net()

    ema = EMA(model)

    ema_param_before = list(ema.model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    optimizer.zero_grad()
    data = torch.randn(2, 10, requires_grad=True)
    out = model(data)
    labels = torch.randint(0, 2, size=(2, 2)).float()
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    ema.step(model)

    ema_param_after = list(ema.model.parameters())

    print("done")
