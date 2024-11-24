#!/bin/bash

#PJM -g gk36
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -N dino_run
#PJM   -j

cd ../../../..

module load aquarius
module load miniconda/py39_4.9.2
module load cuda/11.3
#conda init
source activate pytorch
cd Coop/CoOp-main

# custom config
DATA='Data'
TRAINER=CoOp
# TRAINER=CoOp

DATASET='imagenet_a'
SEED=1

CFG=vit_b16_ep100_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
#DIR=output/imagenet/CoOp/vit_b16_ep100_ctxv1_16shots/nctx16_cscFalse_ctpend/seed${SEED}

if [ -d "$DIR" ]; then
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/CoOp/vit_b16_ep100_ctxv1_16shots/nctx16_cscFalse_ctpend/seed${SEED} \
    --load-epoch 100 \
    --eval-only
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/CoOp/vit_b16_ep100_ctxv1_16shots/nctx16_cscFalse_ctpend/seed${SEED} \
    --load-epoch 100 \
    --eval-only
fi