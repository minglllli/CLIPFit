#!/bin/bash

#PJM -g gk36
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -N clipfit_run
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
TRAINER=ClipFit
# TRAINER=CoOp

DATASET_ALL=('ucf101' 'caltech101' 'food101' 'fgvc_aircraft' 'oxford_pets' 'oxford_flowers' 'eurosat' 'dtd' 'stanford_cars' 'sun397' 'imagenet')
#DATASET=${DATASET_ALL[8]}  #fgvc_aircraft
DATASET='imagenetv2'
CFG=vit_b16_ep100_ctxv1
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16

WEIGHT1=0.5
WEIGHT2=0.2

#DIR=output/imagenet/CoOp/vit_b16_ep100_ctxv1_16shots/nctx16_cscFalse_ctpend/seed${SEED}

for SEED in 1 2 3
do
    DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
    if [ -d "$DIR" ]; then
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/imagenet/ClipFit/vit_b16_ep10_ctxv1_16shots_block10/nctx16_cscFalse_ctpend/seed${SEED} \
        --load-epoch 10 \
        --eval-only\
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2}
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/imagenet/ClipFit/vit_b16_ep10_ctxv1_16shots_block10/nctx16_cscFalse_ctpend/seed${SEED} \
        --load-epoch 10 \
        --eval-only\
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2}
    fi
done