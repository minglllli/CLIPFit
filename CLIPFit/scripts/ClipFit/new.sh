#!/bin/bash
#PJM -g gu14
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

# custom config
DATA='Data/'
TRAINER=ClipFit
DATASET_ALL=('ucf101' 'caltech101' 'food101' 'fgvc_aircraft' 'oxford_pets' 'oxford_flowers' 'eurosat' 'dtd' 'stanford_cars'  'sun397' 'imagenet')
DATASET=${DATASET_ALL[2]}  #fgvc_aircraft
CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
WEIGHT1=0.2
WEIGHT2=0.2

LOADEP=100
SUB=new

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}_alpha0/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    DIR=output/base2new/test_${SUB}/${COMMON_DIR}


    if [ -d "$DIR" ]; then
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done


