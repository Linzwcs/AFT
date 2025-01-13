from openmoa import MoAInstance
from openmoa import Engine
from .basic import Module


class ResponseModule(Module):
    pass


class Sampling(ResponseModule):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine

    def forward(self, x: MoAInstance):
        return [ins[:1] for ins in x.response]


class Exploration(ResponseModule):
    def __init__(self, engine: Engine, n, **gen_kwargs):
        super().__init__()
        self.engine = engine
        self.n = n
        self.kwargs = gen_kwargs

    def forward(
        self,
        x: MoAInstance,
        **kwargs,
    ):
        args = self.kwargs.copy()
        args.update(kwargs)

        return self.engine.produce(
            x,
            n=self.n,
            **args,
        )


class RMBestSampling(Sampling):
    def __init__(
        self,
        engine: Engine,
        topk=1,
    ):
        self.engine = engine
        self.topk = topk

    def forward(self, x):
        x = self.engine.produce(x)
        return x.first_n_response(self.topk)
