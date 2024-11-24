# sh feat_extractor.sh
module load aquarius
module load miniconda/py39_4.9.2
module load cuda/11.3
#conda init
source activate pytorch
DATA='/work/03/gu14/k36105/Coop/CoOp-main/Data/'
OUTPUT='/work/03/gu14/k36105/Coop/CoOp-main/clip_feat/'
SEED=1

# oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech101 ucf101 imagenet
for DATASET in eurosat
do
    for SPLIT in val test
    do
        python feat_extractor.py \
        --split ${SPLIT} \
        --root ${DATA} \
        --seed ${SEED} \
        --dataset-config-file /work/03/gu14/k36105/Coop/CoOp-main//configs/datasets/${DATASET}.yaml \
        --config-file /work/03/gu14/k36105/Coop/CoOp-main/configs/trainers/ClipFit/vit_b16_ep100_ctxv1.yaml \
        --output-dir ${OUTPUT} \
        --eval-only
    done
done
