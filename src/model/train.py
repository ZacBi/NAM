from ast import parse
import os
import torch
import random
import argparse
import numpy as np
from trainer import *
from dataset import *

TASK_TYPES = ['text2image', 'image2image', 'text2video']


def main():
    # Arg Parsing
    parser = argparse.ArgumentParser(
        description="Command line interface for Delta Tuning.")
    parser.add_argument("--model_type", default="LlavaPrompt",
                        type=str, help="type of model")
    parser.add_argument("-p", "--from_pretrained",
                        action="store_true", help="From Pretrained or Not")
    parser.add_argument("--load_backbone", default="", type=str,
                        help="Use customized backbone instead of hugging face provided")
    parser.add_argument("--save_to", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")

    parser.add_argument("--task_type", default="text2image", choices=TASK_TYPES,
                        type=str, help="Trained Task Type")
    parser.add_argument("--data_path", default="guangyil/laion-coco-aesthetic",
                        type=str, help="Path to the data, default laion-coco")

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--random_seed", default=0,
                        type=int, help="random seed used")
    parser.add_argument("--epoch", default=4, type=int, help="how many epochs")
    parser.add_argument("--eval_every_step", default=10,
                        type=int, help="Number of iterations between validation")
    parser.add_argument("--lr", "--learning_rate",
                        default=1e-3, type=float, help="learning rate")
    parser.add_argument("--bz", "--batch", default=8,
                        type=int,  help="batch size")
    parser.add_argument("--early_stop", action="store_true",
                        help="whether apply early stopping")

    parser.add_argument("--encoder_model_pth",
                        default="", help="encoder path")
    parser.add_argument("--generator_model_pth",
                        default="", help="generator path")
    parser.add_argument("--backbone_model_pth",
                        default="", help="backbone path")
    parser.add_argument("--input_projector_type",
                        default="mlp", help="input projector type")
    parser.add_argument("--input_projector_depth", default=1, type=int,
                        help="input projector depth, effective only when input_projector_type is mlp")
    parser.add_argument("--output_projector_type",
                        default="mlp", help="output projector type")
    parser.add_argument("--output_projector_depth", default=1, type=int,
                        help="output projector depth, effective only when output_projector_type is mlp")

    args = parser.parse_args()

    # Process sub args
    # split by 'to'(2)
    task_type = args.task_type.split('2')
    args.input_type = task_type[0]
    args.output_type = task_type[1]

    # 先默认写llama, 后面根据backbone_model_pth来分析
    args.backbone_type = 'llama'

    # Randomness Fixing
    seed = args.random_seed
    seed += hash(args.task_type) % (2**32)
    print("Random seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Data Preperation
    args.task_type = args.task_type.lower()
    train, valid, test = getDataset(args)

    # Preprocessing args
    args.verb = args.verb.split(",")
    if (not os.path.exists(args.save_to)):
        os.makedirs(args.save_to)

    t = trainer(args)
    t.train(train, valid, test, num_train_epochs=args.epoch)


if __name__ == "__main__":
    main()
