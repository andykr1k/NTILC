#!/bin/bash

mkdir -p logs/man

export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup python -m utils.generate_man_nl_command_pairs \
> logs/man/generateTuples.log 2>&1 &
