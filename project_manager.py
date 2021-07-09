import os
import argparse

import container.modeling.config as cfg

def init_s3_buckets():
    os.system(f"aws s3 mb s3://{cfg.PROJECT_NAME}.test")
    # os.system(f"aws s3 mb s3://{cfg.PROJECT_NAME}.data")
    # os.system(f"aws s3 mb s3://{cfg.PROJECT_NAME}.output")
    # os.system(f"aws s3 sync {cfg.DATA_LOCAL_DIR} s3://{cfg.PROJECT_NAME}.data")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_s3_buckets",
        help="Use flag if you want to init the s3 buckets",
        action='store_true')

    args = parser.parse_args() 

    print(args.init_s3_buckets)

    if args.init_s3_buckets:
        init_s3_buckets()
