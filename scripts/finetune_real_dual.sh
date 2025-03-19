#!/usr/bin/env bash
#### Multi-Concept Fine-Tuning Script
# 1st arg: V1* caption (e.g., cartoon_poly_bear)
# 2nd arg: V1* images path (e.g., /home/acorn/custom-diffusion/data/pogny)
# 3rd arg: V2* caption (e.g., cartoon_cat)
# 4th arg: V2* images path (e.g., /home/acorn/custom-diffusion/data/mumu)
# 5th arg: experiment name
# 6th arg: config name
# 7th arg: pretrained model checkpoint path

ARRAY=()

for i in "$@"; do 
    echo $i
    ARRAY+=("$i")
done

# 예시: 두 개의 캡션을 별도로 전달하여,
# V1*에는 "<new1> cartoon_poly_bear", V2*에는 "<new2> cartoon_cat"를 사용한다고 가정
# (train.py가 --caption2, --datapath2, --modifier_token2 등을 지원해야 합니다)

if [ "${ARRAY[5]}" == "finetune.yaml" ]; then
    python -u train.py \
        --base configs/custom-diffusion/${ARRAY[5]} \
        -t --gpus ,1 \
        --resume-from-checkpoint-custom ${ARRAY[6]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --caption2 "<new2> ${ARRAY[2]}" \
        --datapath2 ${ARRAY[3]} \
        --reg_datapath "" \
        --reg_caption "" \
        --name "${ARRAY[4]}-sdv4"
else
    python -u train.py \
        --base configs/custom-diffusion/${ARRAY[5]} \
        -t --gpus ,1 \
        --resume-from-checkpoint-custom ${ARRAY[6]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --caption2 "<new2> ${ARRAY[2]}" \
        --datapath2 ${ARRAY[3]} \
        --modifier_token "<new1>" \
        --modifier_token2 "<new2>" \
        --name "${ARRAY[4]}-sdv4"
fi
