import json
from pathlib import Path
from argparse import ArgumentParser


from transformers import AutoTokenizer
from easydict import EasyDict

def bioasq_test_preprocessing(cfg:EasyDict, test_file_path:str):
    # setup for preprocessing
    print(f"Preprocessing {test_file_path}")
    test_file = Path(test_file_path).open()
    test_dataset = json.load(test_file)
    save_file = Path(Path(test_file_path).name).open("w")
    
    print(f"{cfg.model.encoder_name} tokenizer loading...", end="")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_name)
    print("END")

    for line in test_dataset["documents"]:
        title = line["title"]
        abs = line["abstractText"]
        did = line["pmid"]
        text = " ".join([title, abs])
        input_ids = tokenizer(
                                text,
                                max_length=cfg.data.max_seq_length,
                                truncation=True
                                )["input_ids"]
        instance = {"did" : did,
                    "input_ids" : input_ids}
        save_file.write(json.dumps(instance)+"\n")

def main():
    parser = ArgumentParser()
    # general args
    parser.add_argument("--config", type=str, default=None, required=True,
                        help="Dataset name")
    parser.add_argument("--testfile", type=str, default=None, required=True,
                        help="Dataset name")
    
    args = parser.parse_args()

    cfg = EasyDict(json.load(Path(args.config).open()))

    bioasq_test_preprocessing(cfg, args.testfile)


if __name__ == "__main__":
    main()
