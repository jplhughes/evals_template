from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from evals.run import main as run_main


def sweep():
    GlobalHydra.instance().clear()
    # config_path must be relative to this file
    with initialize(config_path="../evals/conf/"):
        cfg = compose(config_name="config")
        # load in yaml file for prompt
        cot_prompt_conf = OmegaConf.load("./evals/conf/prompt/cot.yaml")
        OmegaConf.update(cfg, "prompt", cot_prompt_conf)
        OmegaConf.update(cfg, "limit", 5)

        for model in ["gpt-3.5-turbo", "claude-2.1"]:
            for temperature in [0.0, 0.3, 0.6]:
                cfg.language_model.model = model
                cfg.language_model.temperature = temperature
                OmegaConf.update(cfg, "exp_dir", f"./exp/sweep/{model}/{temperature}")
                print(OmegaConf.to_yaml(cfg))

                run_main(cfg)


if __name__ == "__main__":
    sweep()
