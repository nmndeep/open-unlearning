import copy
import torch
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_js_avg, compute_kl_divergence, compute_js_models, jslossnaman, compute_js_models_naman, compute_js_matthias


class GradJSDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha #1-self.gamma
        self.retain_loss_type = retain_loss_type
        self.ref_model = None

        if retain_loss_type == "KL" or retain_loss_type == "JS" or retain_loss_type=='JSNaman':
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        elif self.retain_loss_type == "JS":
            js_loss, retain_outputs = compute_js_models(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += js_loss
        elif self.retain_loss_type == "JSNaman":
            js_loss, retain_outputs = compute_js_models_naman(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += js_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss


    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        if self.retain_loss_type == "JSNaman": # need to fix this -for now, use same identifier for both
            print("Using JSNaman")
            forget_loss, forget_outputs = jslossnaman(model, forget_inputs)
        else:
            forget_loss, forget_outputs = compute_js_avg(model, forget_inputs)
        # forget_outputs = model(**forget_inputs)
        # forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss

# a version of GradJSDiff that uses the KL divergence as the retain loss per default
class GradJSDiffKL(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        # Set retain_loss_type to "KL" by default
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="KL", *args, **kwargs)

class GradJSDiffJS(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        # Set retain_loss_type to "JS" by default
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="JS", *args, **kwargs)

class GradJSAscentWithRetainLogging(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, *args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_js_avg(model, forget_inputs)
        # forget_outputs = model(**forget_inputs)
        # forget_loss = -forget_outputs.loss
        with torch.no_grad():
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = forget_loss
        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss

class GradJSAscentNamanWithRetainLogging(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type=retain_loss_type, *args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = js_loss(model, forget_inputs)
        # forget_outputs = model(**forget_inputs)
        # forget_loss = -forget_outputs.loss
        with torch.no_grad():
            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = forget_loss
        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss

class GradJSDiffJSNaman(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        # Set retain_loss_type to "JS" by default
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="JSNaman", *args, **kwargs)

class GradJSDiffJSMatthias(GradJSDiff):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        # Set retain_loss_type to "JS" by default
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type="JSNaman", *args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        # compute the JS loss to uniform, but set relevant tokens to zero
        forget_loss, forget_outputs = compute_js_matthias(model, forget_inputs)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss