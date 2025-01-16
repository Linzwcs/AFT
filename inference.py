from aft.utils import load_MoA_model
from aft.aft import MoAInstance
from aft.config import Config
import argparse
import json


# Function to load data from a JSONL file
def load_jsonl(jsonl_file_path):
    jsonl_data = []
    with open(jsonl_file_path, "r") as jsonl_file:
        for line in jsonl_file:
            jsonl_data.append(json.loads(line.strip()))
    return jsonl_data


# Function to save data to a JSON file or JSONL file
def save_jsonl(data, json_file_path, is_jsonl=False):
    with open(json_file_path, "w") as json_file:
        if is_jsonl:
            # Save as JSONL format (one object per line)
            for item in data:
                json_file.write(json.dumps(item) + "\n")
        else:
            # Save as standard JSON format (list of objects)
            json.dump(data, json_file, indent=4)


# 创建解析器
def main():
    parser = argparse.ArgumentParser(
        description="inference code Sample of Self Mixture of Experts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="path to config",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/alpaca_eval_v2.json",
        help="path to input data with jsonl format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/alpaca_eval_v2.json",
        help="output path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size of generation",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.load_yaml(f)

    model = load_MoA_model(config, "vllm")
    data = load_jsonl(args.input)
    bsz = args.batch_size

    for idx in range(0, len(data), bsz):
        start, end = idx, idx + bsz
        querys = []
        for item in data[start:end]:
            querys.append(item["query"])
        x = MoAInstance(
            context=querys,
            sys=None,
            response=None,
        )
        output = model(x)
        response = [r[0] for r in output.response]
        for item, r in zip(data[start:end], response):
            item["response"] = r

    save_jsonl(data, args.output, False)


if __name__ == "__main__":
    main()
