from utils.common_util import get_args
from stage1 import stage1
from stage2 import stage2

if __name__ == "__main__":
    args=get_args()
    if args.stage=="1":
        stage1(args)
    if args.stage=="2":
        stage2(args)
    # to do stage3