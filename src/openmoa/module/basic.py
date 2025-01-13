from ..openmoa import MoAInstance
from ..infer_engine import Engine


class Module:
    def __init__(self):
        pass

    def forward(self, x: MoAInstance, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: MoAInstance) -> MoAInstance:
        for m in self.modules:
            x = m(x)
        return x
