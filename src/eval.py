
import hydra
from omegaconf import DictConfig

from evals import get_evaluators
from model import get_model
from trainer.utils import seed_everything


def change_at_runtime(eval_cfg, typ):

    # print(OmegaConf.to_yaml(eval_cfg))
    if isinstance(eval_cfg, DictConfig):
        for k, v in eval_cfg.items():
            # If there's an args section with question_key, update it
            if (
                isinstance(v, DictConfig)
                and "args" in v
                and isinstance(v.args, DictConfig)
                and "question_key" in v.args
            ):
                v.args.question_key = typ
            else:
                # Recursively check nested configs
                change_at_runtime(v, typ)
    # keys = ['TOFU_QA_forget', 'TOFU_QA_forget_para', 'TOFU_QA_forget_pert']
    # for key in keys:
    #     key_path = f"tofu.metrics.forget_quality.forget_Q_A_Prob.datasets.{key}.args.question_key"
    #     OmegaConf.update(eval_cfg, key_path, typ)
    # return eval_cfg


# def update_question_keys(cfg_section, new_key):
#     if isinstance(cfg_section, DictConfig):
#         for k, v in cfg_section.items():
#             # If there's an args section with question_key, update it
#             if (
#                 isinstance(v, DictConfig)
#                 and "args" in v
#                 and isinstance(v.args, DictConfig)
#                 and "question_key" in v.args
#             ):
#                 v.args.question_key = new_key
#             else:
#                 # Recursively check nested configs
#                 update_question_keys(v, new_key)

    


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    # exit()
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    
    # Change the type of question dynaimcally to test for? Can be done cleaner: DUDE worry later
    change_at_runtime(eval_cfgs.tofu.metrics.forget_quality, cfg.qtype)

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
