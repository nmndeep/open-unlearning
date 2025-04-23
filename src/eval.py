import hydra
from omegaconf import DictConfig

from model import get_model
from evals import get_evaluators

# def change_at_runtime(eval_cfg, suffix):

#     # print(OmegaConf.to_yaml(eval_cfg))
#     if isinstance(eval_cfg, DictConfig):
#         for k, v in eval_cfg.items():
#             # If there's an args section with question_key, update it
#             if (
#                 isinstance(v, DictConfig)
#                 and "args" in v
#                 and isinstance(v.args, DictConfig)
#                 and "answer_key" in v.args
#             ):
#                 v.args.answer_key = v.args.answer_key+suffix
#             else:
#                 # Recursively check nested configs
#                 change_at_runtime(v, suffix)

def change_at_runtime(eval_cfg, suffix):

    # print(OmegaConf.to_yaml(eval_cfg))
    if isinstance(eval_cfg, DictConfig):
        for k, v in eval_cfg.items():
            # If there's an args section with question_key, update it
            if (
                isinstance(v, DictConfig)
                and "args" in v
                and isinstance(v.args, DictConfig)
                and "answer_key" in v.args
            ):
                v.args.answer_key = v.args.answer_key+suffix
                v.args.question_key = v.args.question_key+suffix # ugly fix
            else:
                # Recursively check nested configs
                change_at_runtime(v, suffix)

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    if 'tofu' in eval_cfgs.keys():
        pass
        change_at_runtime(eval_cfgs.tofu, '_real')
        # change_at_runtime(eval_cfgs.tofu.metrics.forget_Q_A_Prob, '_short')
        # change_at_runtime(eval_cfgs.tofu.metrics.forget_Q_A_ROUGE, '_short')
        # eval_cfgs.tofu.metrics.forget_quality.args.answer_key = 'answer_short'
    else:
        change_at_runtime(eval_cfgs.muse.metrics.forget_knowmem_ROUGE, 'answer_short')
    from omegaconf import OmegaConf
    import yaml

    with open('tmp-config-2.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(eval_cfgs), f)

    print("This is a test")
    # print(eval_cfgs)
    # print(eval_cfgs['tofu']['metrics']['forget_quality'].keys())
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
