from vllm import LLM, SamplingParams
from openmoa import MoAInstance
from .base_engine import Engine


class VllmGenEngine(Engine):
    def __init__(self, path: str, vllm_args: dict = dict()):
        super().__init__()
        self.vllm_args = vllm_args
        self.llm = LLM(path, **vllm_args)
        self.tokenizer = self.llm.get_tokenizer()
        self.default_stop_ids = (self.tokenizer.eos_token_id,)

    def produce(self, x, n, **kwargs):
        if kwargs.get("stop_token_ids", None) is None:
            kwargs["stop_token_ids"] = self.default_stop_ids
        conversations = x.get_conversations()
        batch_size = len(conversations)
        conversations *= n

        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=SamplingParams(n=1, **kwargs),
        )
        assert len(outputs) == n * batch_size
        assert len(outputs[0].outputs) == 1
        response = []
        for i in range(batch_size):
            res = []
            for j in range(n):
                res.append(outputs[i + j * batch_size].outputs[0].text)
            response += [res]
        return MoAInstance(
            context=x.context,
            sys=x.sys,
            response=response,
        )
