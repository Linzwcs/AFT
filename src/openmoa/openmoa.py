class MoAInstance:
    sys: list[str]
    context: list[list[dict]]
    response: list[list[str]]

    def __init__(self, context: list[dict], sys: str, response: list[str]):
        self.context = context
        self.sys = sys
        self.response = response

    def first_n_response(self, n):
        return [ins[:n] for ins in self.response]

    def get_conversations(self):
        ret = []
        for s, c in zip(self.sys, self.context):
            ret.append([{"role": "system", "content": s}] + c)
        return ret
