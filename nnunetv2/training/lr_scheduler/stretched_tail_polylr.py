from torch.optim.lr_scheduler import _LRScheduler


class StretchedTailPolyLRScheduler(_LRScheduler):
    """
    Epochs [0, k_transition): same LR as PolyLR with max_steps=ref_poly_steps (default nnUNet-style poly).
    Epochs [k_transition, n_last]: linear LR from poly(k_transition) to poly(ref_poly_steps-1), stretched.

    k_transition and the poly/tail logic use effective epoch index s = max(0, global_epoch - epoch_offset).
    Use epoch_offset=warmup_duration (e.g. with nnUNetTrainer_warmup) so post-warmup epoch 0 matches poly head start.
    """

    def __init__(
        self,
        optimizer,
        initial_lr: float,
        num_epochs: int,
        k_transition: int = 750,
        ref_poly_steps: int = 1000,
        exponent: float = 0.9,
        last_epoch: int = None,
        epoch_offset: int = 0,
    ):
        self.initial_lr = initial_lr
        self.num_epochs = num_epochs
        self.k_transition = k_transition
        self.ref_poly_steps = ref_poly_steps
        self.exponent = exponent
        self.epoch_offset = epoch_offset
        self.ctr = 0
        if not (0 < k_transition < ref_poly_steps):
            raise ValueError(f"require 0 < k_transition < ref_poly_steps; got k={k_transition}, ref={ref_poly_steps}")
        if not (0 <= epoch_offset < num_epochs):
            raise ValueError(f"require 0 <= epoch_offset < num_epochs; got offset={epoch_offset}, num_epochs={num_epochs}")
        super().__init__(optimizer, last_epoch if last_epoch is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        s = max(0, current_step - self.epoch_offset)
        n_last = (self.num_epochs - 1) - self.epoch_offset
        ref = self.ref_poly_steps
        exp = self.exponent
        lr0 = self.initial_lr
        k = self.k_transition

        if s < k or n_last <= k:
            new_lr = lr0 * (1.0 - s / ref) ** exp
        else:
            lr_k = lr0 * (1.0 - k / ref) ** exp
            lr_end = lr0 * (1.0 - (ref - 1) / ref) ** exp
            denom = max(n_last - k, 1)
            t = (s - k) / denom
            new_lr = lr_k + (lr_end - lr_k) * t

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr
