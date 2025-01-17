from .infer_engine import vllm
from .config import Config
from .net import MoA


def load_MoA_model(
    config: Config,
    backend: str,
):
    if config.vllm_seed is None:
        engine = vllm.VllmGenEngine(
            path=config.model_name,
            vllm_args=dict(),
        )
    else:
        engine = vllm.VllmGenEngine(
            path=config.model_name,
            vllm_args=dict(seed=config.vllm_seed),
        )
    if backend == "vllm":
        model = MoA(
            engine=engine,
            gen_configs=config.generation_params,
            num_aggregation=config.num_aggregation,
            # final_agg=config.final_aggregation,
        )
    else:
        raise NotImplementedError
    return model
