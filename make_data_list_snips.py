"""Script for creating Google Speech Commands reduced label dataset"""
from argparse import ArgumentParser
from utils.dataset_snips import get_train_val_test_split
import os


def main(args):
    pretrain, train, val, test = get_train_val_test_split(args.data_root,
                                                          args.train_path,
                                                          args.val_path,
                                                          args.test_path,
                                                          args.pretrain)
    if len(pretrain["data"]) > 0:
        with open(os.path.join(args.out_dir, "pretraining_list.txt"), "w+") as f:
            f.write("\n".join(pretrain["data"]))
        with open(os.path.join(args.out_dir, "pretraining_labels.txt"), "w+") as f:
            f.write("\n".join(str(label) for label in pretrain["labels"]))

    with open(os.path.join(args.out_dir, "training_list.txt"), "w+") as f:
        f.write("\n".join(train["data"]))
    with open(os.path.join(args.out_dir, "training_labels.txt"), "w+") as f:
        f.write("\n".join(str(label) for label in train["labels"]))

    with open(os.path.join(args.out_dir, "validation_list.txt"), "w+") as f:
        f.write("\n".join(val["data"]))
    with open(os.path.join(args.out_dir, "validation_labels.txt"), "w+") as f:
        f.write("\n".join(str(label) for label in val["labels"]))

    with open(os.path.join(args.out_dir, "testing_list.txt"), "w+") as f:
        f.write("\n".join(test["data"]))
    with open(os.path.join(args.out_dir, "testing_labels.txt"), "w+") as f:
        f.write("\n".join(str(label) for label in test["labels"]))

    print("Saved data path lists and labels.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-tr", "--train_path", type=str, required=True, help="Path to train.json.")
    parser.add_argument("-v", "--val_path", type=str, required=True, help="Path to train.json.")
    parser.add_argument("-t", "--test_path", type=str, required=True, help="Path to dev.json.")
    parser.add_argument("-d", "--data_root", type=str, required=True,
                        help="Root directory of Snips dataset.")
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="Output directory for data lists and labels.")
    parser.add_argument("--pretrain", type=float, required=False, default=None,
                        help="Amount of train set reserved for pretraining, e.g., 0.8 for 80%.")
    args = parser.parse_args()

    main(args)
