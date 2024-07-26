from argparse import ArgumentParser

from rcv1v2_preprocessor import RCV1v2Preprocessor
from nyt_preprocessor import NYTPreprocessor
from eurlex57k_preprocessor import EURLEX57KPreprocessor

preprocessors = {
    "RCV1v2" : RCV1v2Preprocessor,
    "NYT" : NYTPreprocessor,
    "EURLEX57K" : EURLEX57KPreprocessor
}

def main():
    parser = ArgumentParser()
    # general args
    parser.add_argument("--name", type=str, default=None, required=True,
                        help="dataset name")
    parser.add_argument("--raw_dir", type=str, default=None, required=True,
                        help="raw data dir")
    parser.add_argument("--save_dir", type=str, default=None, required=True,
                        help="save dir")
    parser.add_argument("--hierarchy_file", type=str, default=None, required=False,
                        help="EURLEX57K.json file path")
    
    
    args = parser.parse_args()

    preprocessor = preprocessors[args.name](args.raw_dir, args.save_dir, args.hierarchy_file)

    preprocessor.preprocessing()

if __name__ == "__main__":
    main()

# python src/preprocess.py --name=RCV1v2 --raw_dir=[raw_data_dir] --save_dir=[save_dir]
# python src/preprocess.py --name=NYT --raw_dir=[raw_data_dir] --save_dir=[save_dir]
# python src/preprocess.py --name=EURLEX57K --raw_dir=[raw_data_dir] --save_dir=[save_dir]  --hierarchy_file=[EURLEX57K.json_file path]