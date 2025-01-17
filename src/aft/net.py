from . import (
    ReformMoudle,
    Exploration,
    MoAInstance,
    Module,
    Sequential,
    GenerationArgs,
)


class AggReform(ReformMoudle):
    def forward(self, x: MoAInstance):

        template = "You are a helpful aggregator.\nYou have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n    \n    Responses from models:"
        ret_sys = []
        for _, r in zip(x.sys, x.response):
            ret_sys.append(
                template + "\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(r)])
            )
        return MoAInstance(context=x.context, sys=ret_sys, response=None)


class ProposorReform(ReformMoudle):
    def forward(self, x: MoAInstance):
        return MoAInstance(
            context=x.context,
            sys=["\n"] * len(x.context),
            response=None,
        )


class AggLayer(Module):
    def __init__(self, engine, n, **gen_kwargs):
        self.agg_reform = AggReform()
        self.explore = Exploration(engine, n=n, **gen_kwargs)

    def forward(self, x, **kwargs):
        x = self.agg_reform(x)
        x = self.explore(x, **kwargs)
        return x


class ProposorLayer(Module):
    def __init__(self, engine, n, **gen_kwargs):
        self.reform = ProposorReform()
        self.explore = Exploration(engine, n=n, **gen_kwargs)

    def forward(self, x, **kwargs):
        x = self.reform(x)
        x = self.explore(x, **kwargs)
        return x


class MoA(Module):
    def __init__(
        self,
        engine,
        gen_configs: GenerationArgs,
        num_aggregation=2,
        # final_agg=True,
    ):
        super().__init__()
        self.engine = engine
        self.proposor = ProposorLayer(
            engine=engine,
            **gen_configs.gen_config(),
        )
        self.aggs = Sequential(
            *(
                AggLayer(
                    engine,
                    **gen_configs.gen_config(),
                )
                for i in range(num_aggregation - 1)
            )
        )
        final_agg_params = gen_configs.final_gen_config()
        final_agg_params["n"] = 1
        self.head = AggLayer(engine, **final_agg_params)

    def forward(self, x, *args, **kwargs):
        x = self.proposor(x)
        x = self.aggs(x)
        x = self.head(x, *args, **kwargs)
        return x
