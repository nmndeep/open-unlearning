import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs

def compute_js_models(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # Compute the midpoint distribution
    m = 0.5 * (ref_probs + current_probs)

    # js divergence
    kl_p_m = F.kl_div(m.log(), ref_probs, reduction='batchmean')
    kl_q_m = F.kl_div(m.log(), current_probs, reduction='batchmean')

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

def compute_js_avg(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch

    # Compute probabilities from logits
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
        
    # Define the uniform distribution over the vocabulary
    uniform_dist = torch.full_like(probs, 1.0 / probs.size(-1))

    # Compute the midpoint distribution
    m = 0.5 * (probs + uniform_dist)

    # Compute KL divergence (adding a small value to avoid log(0))
    kl_p_m = F.kl_div(m.log(), probs, reduction='batchmean')
    kl_q_m = F.kl_div(m.log(), uniform_dist, reduction='batchmean')

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

def jslossnaman(model, forget_inputs):
    outputs = model(**forget_inputs)
    logits = outputs.logits
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    uniform_dist = torch.full_like(probs, 1.0 / probs.size(-1))

    # Midpoint distribution
    m = 0.5 * (probs + uniform_dist)
    m = torch.clamp(m, min=epsilon)  # Ensure no zeros

    # Clamp input distributions as well to avoid log(0)
    probs = torch.clamp(probs, min=epsilon)
    uniform_dist = torch.clamp(uniform_dist, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(uniform_dist * (torch.log(uniform_dist) - torch.log(m)), dim=-1).mean()

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

def jslossnamantoken11(model, forget_inputs, token_id=11):
    outputs = model(**forget_inputs)
    logits = outputs.logits
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    vocab_size = logits.size(-1)
    peaked_dist = torch.zeros_like(probs)
    peaked_token = token_id
    if peaked_token >= vocab_size:
        raise ValueError(f"Token {peaked_token} is out of range for vocab size {vocab_size}")

    peaked_dist[..., peaked_token] = 1.0  # Set probability 1.0 at token 11 for all positions

    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)
    m = torch.clamp(m, min=epsilon)  # Ensure no zeros

    # Clamp input distributions as well to avoid log(0)
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1).mean()

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs


def compute_js_models_naman(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_logits = ref_outputs.logits
    epsilon = 1e-9
    ref_probs = F.softmax(ref_logits, dim=-1)



    outputs = model(**inputs)
    logits = outputs.logits
    current_probs = F.softmax(logits, dim=-1)

    # Compute the midpoint distribution
    m = 0.5 * (ref_probs + current_probs)

    # Clamp input distributions as well to avoid log(0)
    ref_probs = torch.clamp(ref_probs, min=epsilon)
    current_probs = torch.clamp(current_probs, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(ref_probs * (torch.log(ref_probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(current_probs * (torch.log(current_probs) - torch.log(m)), dim=-1).mean()

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

def compute_uniform_ce_avg(model, inputs):
    # Forward pass to get logits
    outputs = model(**inputs)
    logits = outputs.logits

    # Compute log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Define the uniform target distribution over the vocabulary
    vocab_size = logits.size(-1)
    uniform_dist = torch.full_like(log_probs, 1.0 / vocab_size)

    # Compute cross-entropy loss: CE(target, log_probs)
    ce_loss = -(uniform_dist * log_probs).sum(dim=-1)  # sum over vocab
    ce_loss = ce_loss.mean()  # average over batch and sequence if needed

    return ce_loss, outputs



def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_js_avg_token11_peak(model, inputs):
    # Compute model output
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
    probs = F.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)

    batch_size, seq_len, vocab_size = probs.shape

    # Create a distribution peaked at token 11
    peaked_dist = torch.zeros_like(probs)
    peaked_token = 11
    if peaked_token >= vocab_size:
        raise ValueError(f"Token {peaked_token} is out of range for vocab size {vocab_size}")

    peaked_dist[..., peaked_token] = 1.0  # Set probability 1.0 at token 11 for all positions

    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)

    # Compute JS divergence: average of KL divergences from probs and peaked_dist to m
    eps = 1e-8
    m_log = torch.log(m + eps)

    kl_p_m = F.kl_div(m_log, probs, reduction='batchmean')
    kl_q_m = F.kl_div(m_log, peaked_dist, reduction='batchmean')
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

import random

def compute_js_avg_random_peak(model, inputs):
    # Compute model output
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
    probs = F.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)

    batch_size, seq_len, vocab_size = probs.shape

    # Create a "peaked" distribution: one-hot at a random token for each position
    peaked_dist = torch.zeros_like(probs)
    for b in range(batch_size):
        for t in range(seq_len):
            random_token = random.randint(0, vocab_size - 1)
            peaked_dist[b, t, random_token] = 1.0

    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)

    # Compute JS divergence: average of KL divergences from probs and peaked_dist to m
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    m_log = torch.log(m + eps)

    kl_p_m = F.kl_div(m_log, probs, reduction='batchmean')
    kl_q_m = F.kl_div(m_log, peaked_dist, reduction='batchmean')
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

def jslossnamanrandtoken(model, forget_inputs):
    outputs = model(**forget_inputs)
    logits = outputs.logits
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    vocab_size = logits.size(-1)
    peaked_dist = torch.zeros_like(probs)
    # for b in range(batch_size):
    #     for t in range(seq_len):
    #         random_token = random.randint(0, vocab_size - 1)
    #         peaked_dist[b, t, random_token] = 1.0

    # peaked_token = 11
    # if peaked_token >= vocab_size:
    #     raise ValueError(f"Token {peaked_token} is out of range for vocab size {vocab_size}")
    random_token = random.randint(0, vocab_size - 1)

    peaked_dist[..., random_token] = 1.0  # Set probability 1.0 at token 11 for all positions


    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)
    m = torch.clamp(m, min=epsilon)  # Ensure no zeros

    # Clamp input distributions as well to avoid log(0)
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1).mean()

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs


def compute_js_matthias(model, forget_inputs):
    outputs = model(**forget_inputs)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # Get the set of target tokens from labels
    labels = forget_inputs["labels"]  # (batch_size, seq_len)
    # print(labels.shape,labels)
    
    # Create a mask over vocab where tokens in labels are excluded (set to 0)
    vocab_size = logits.size(-1)
    # print('vocab size',vocab_size)
    batch_size, seq_len = labels.size()
    
    # Build a binary mask for tokens to be zeroed out in the uniform distribution
    mask = torch.ones((batch_size, seq_len, vocab_size), device=logits.device)

    # For each position in the batch and sequence, zero out the corresponding label token
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = labels[b, s].item()
            if token_id != -100:  # -100 is used for ignored positions in loss
                mask[b, s, token_id] = 0.0

    # Create a uniform distribution over the remaining tokens
    uniform_zero = mask / mask.sum(dim=-1, keepdim=True).clamp(min=epsilon)  # Avoid divide-by-zero
    uniform_zero = torch.clamp(uniform_zero, min=epsilon)  # To prevent log(0)

    # Midpoint distribution
    m = 0.5 * (probs + uniform_zero)
    m = torch.clamp(m, min=epsilon)

    # Clamp input distributions
    probs = torch.clamp(probs, min=epsilon)

    # Compute KL divergences
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(uniform_zero * (torch.log(uniform_zero) - torch.log(m)), dim=-1).mean()

    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs
