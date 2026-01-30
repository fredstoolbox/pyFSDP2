# pyFSDP2
This example training script uses pytorch's Fully Sharded Data Parallel to train a simple conv network. The script is meant to illustrate the basic steps for setting up FSDP pipeline to train custom model on multi-node, multi-GPUs.

Link to pytorch's FSDP2 page: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html

Link to the FSDP paper: https://arxiv.org/pdf/2304.11277

to run it use torchrun command: torchrun --nproc_per_node 2 ./pyFSDP2/fsdp2_basic.py --save-every=20 --batch-size=10

where --nproc_per_node specifies how many GPUs do you wanna use in the training.
