from typing import Optional, Union
from dataclasses import dataclass
from simple_parsing import Serializable


@dataclass
class AgentArgs(Serializable):
    temperature: float
    top_p: float
    max_tokens: int
    n: int

    def dict(self):
        return dict(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # seed=self.seed,
            n=self.n,
        )


@dataclass
class Config(Serializable):
    model_name: str
    proposal_params: AgentArgs
    aggregation_params: AgentArgs
    # hidden_layer: int
    num_aggregation: int
    vllm_seed: Optional[int] = None
