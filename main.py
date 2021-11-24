import argparse
from stage1 import stage1
from stage2 import stage2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--stage", type=str, default="1", help="select stage")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args=get_args()
    if args.stage=="1":
        stage1(args)
    # if args.stage=="2":
    # to do
    # stage2(args)
    # to do stage3