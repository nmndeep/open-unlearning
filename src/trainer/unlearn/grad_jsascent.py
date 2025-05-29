import copy

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_js_avg, compute_kl_divergence


class GradJSAscent(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma=0.2


    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_js_avg(model, forget_inputs)
        loss = self.gamma * forget_loss
        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": 0,
                "forget_loss": forget_loss.item(),
            })
        return (loss, forget_outputs) if return_outputs else loss
