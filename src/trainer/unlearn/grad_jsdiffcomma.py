import copy
import torch
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_js_avg_token11_peak, compute_kl_divergence, compute_js_models, compute_js_models_naman, jslossnamantoken11


class GradJSDiffComma(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", token_id=11, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha #1-self.gamma
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        self.token_id = token_id

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
        if self.retain_loss_type == "JSNaman":
            forget_loss, forget_outputs = jslossnamantoken11(model, forget_inputs, token_id=self.token_id) # token 11 = comma
        else:
            forget_loss, forget_outputs = compute_js_avg_token11_peak(model, forget_inputs) # token 11 = comma
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

class GradJSDiffCommaNaman(GradJSDiffComma):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JSNaman", token_id=11, *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JSNaman', token_id=11, *args, **kwargs)

class GradJSDiffHashNaman(GradJSDiffComma):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JSNaman",token_id=2,  *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JSNaman', token_id=2, *args, **kwargs)

class GradJSDiffTagsNaman(GradJSDiffComma):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JSNaman", token_id=16309, *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JSNaman', token_id=16309, *args, **kwargs)

class GradJSDiffAndNaman(GradJSDiffComma):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JSNaman", token_id=438, *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JSNaman', token_id=438, *args, **kwargs)

class GradJSAscentCommaNaman(GradJSDiffComma):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='NLL', *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = jslossnamantoken11(model, forget_inputs)
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

# class GradJSAscentCommaNaman(GradJSDiffComma):
#     def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
#         super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='NLL', *args, **kwargs)

#     def compute_loss(self, model, inputs, return_outputs=False):
#         forget_inputs = inputs["forget"]
#         forget_inputs = {
#             "input_ids": forget_inputs["input_ids"],
#             "attention_mask": forget_inputs["attention_mask"],
#             "labels": forget_inputs["labels"],
#         }
#         forget_loss, forget_outputs = compute_js_avg_token11_peak(model, forget_inputs)
#         with torch.no_grad():
#             retain_inputs = inputs["retain"]
#             retain_inputs = {
#                 "input_ids": retain_inputs["input_ids"],
#                 "attention_mask": retain_inputs["attention_mask"],
#                 "labels": retain_inputs["labels"],
#             }
#             retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

#         loss = forget_loss
        
#         if self.accelerator.is_local_main_process:
#             self.log({
#                 "retain_loss": retain_loss.item(),
#                 "forget_loss": forget_loss.item(),
#             })

#         return (loss, forget_outputs) if return_outputs else loss

