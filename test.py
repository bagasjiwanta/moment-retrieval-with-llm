import json
import re


def json_dumps(data, indent=2, max_inline_length=120):
    json_str = json.dumps(data, indent=indent)
    # regex to find lists with no nested braces/brackets
    pattern = re.compile(r"\[\s*([^\[\]\{\}]+?)\s*\]", re.DOTALL)

    def replacer(match):
        content = match.group(1)
        # remove whitespace and newlines inside the list
        inline = " ".join(content.split())
        if len(inline) <= max_inline_length:
            return f"[ {inline} ]"
        else:
            return match.group(0)

    return pattern.sub(replacer, json_str)


def main():
    for split in ["train", "test", "val"]:
        with open(f"datasets/qvhighlights/annotations/processed/highlight_{split}_release.json", "r") as f_in:
            data = json.load(f_in)
            new_data = []
            for d in range(len(data)):
                new_data.append(data[d][0])
            with open(f"datasets/qvhighlights/annotations/processed/highlight_{split}_release2.json", "w") as f_out:
                f_out.write(json_dumps(new_data, indent=2))


if __name__ == "__main__":
    main()
