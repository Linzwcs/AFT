from openmoa.utils import load_MoA_model
from openmoa.openmoa import MoAInstance
from openmoa.config import Config
import argparse


# 创建解析器
def main():
    parser = argparse.ArgumentParser(
        description="inference code Sample of Self Mixture of Experts"
    )

    # 添加参数
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="path to config",
    )
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = Config.load_yaml(f)
    model = load_MoA_model(config, "vllm")
    x = MoAInstance(
        context=[
            [
                {
                    "role": "user",
                    "content": "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for"
                    "her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
                    "How much in dollars does she make every day at the farmers' market?\nLet's think step by step\nAnswer:",
                },
            ]
        ]
        * 2,
        sys=["you are a helpful assistant."] * 2,
        response=None,
    )
    output = model(x)
    print(output.response)


if __name__ == "__main__":
    main()
