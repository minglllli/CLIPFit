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
DATA='Data/'
TRAINER=ClipFit

# test
DATASET_ALL=('ucf101' 'caltech101' 'food101' 'fgvc_aircraft' 'oxford_pets' 'oxford_flowers' 'eurosat' 'dtd' 'stanford_cars'  'sun397' 'imagenet')
DATASET=${DATASET_ALL[7]}
#0,1,3,4,5,6,8 
CFG='vit_b16_ep100_ctxv1'  # config file
CTP='end'  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=4  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

WEIGHT1=0.5
WEIGHT2=0.2



for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        #echo "Oops! The results exist at ${DIR} (so skip this job)"
         python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.W1 ${WEIGHT1} \
        TRAINER.COOP.W2 ${WEIGHT2} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done