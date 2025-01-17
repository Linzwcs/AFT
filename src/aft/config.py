from typing import Optional, Union
from dataclasses import dataclass
from simple_parsing import Serializable


@dataclass
class GenerationArgs(Serializable):
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    final_layer_temperature: float
    final_layer_top_p: float

    def gen_config(self):
        return dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # seed=self.seed,
            n=self.n,
        )

    def final_gen_config(self):
        return dict(
            temperature=self.final_layer_temperature,
            top_p=self.final_layer_top_p,
            max_tokens=self.max_tokens,
            # seed=self.seed,
            n=self.n,
        )


@dataclass
class Config(Serializable):
    model_name: str
    generation_params: GenerationArgs

    num_aggregation: int
    vllm_seed: Optional[int] = None
