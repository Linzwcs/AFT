from .infer_engine import vllm
from .config import Config
from .net import MoA


def load_MoA_model(
    config: Config,
    backend: str,
):
    engine = vllm.VllmGenEngine(
        path=config.model_name,
        vllm_args=dict(seed=config.vllm_seed),
    )
    if backend == "vllm":
        model = MoA(
            engine=engine,
            prop_configs=config.proposal_params,
            agg_configs=config.aggregation_params,
            hidden_layer=config.hidden_layer,
            final_agg=config.final_aggregation,
        )
    else:
        raise NotImplementedError
    return model
