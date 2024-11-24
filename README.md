Pytorch implementation of Paper: Vision-Language Model Fine-Tuning via Simple Parameter-Efficient Modification (EMNLP 2024 Main)

All the experiments are able to run on an A100 GPU.

# Installation Preparation. 
Our code is built on CoOp, so please follow [CoOp](https://github.com/KaiyangZhou/CoOp) to install the Dassl toolbox, package requirements, and datasets. 

# Run Code
## Base2New
The script scripts/ClipFit/base2new_train.sh fine-tunes CLIP on base classes, and the script scripts/ClipFit/new.sh evaluates the performance of fine-tuned CLIP on new classes. Please follow the hyper-parameters reported in the paper. lammbda=8 for all datasets. epoch=10 for Food-101 and ImageNet, and epoch=100 for all other dataaset. Revise CFG in base2new_train.sh for different training scripts, DATASET in base2new_train.sh for different datasets and lambda_ in ClipFit.py. Moreover, please revise the running script to your Linux system forms. 
```
bash base2new_train.sh 
bash new.sh 
```
When the evaluation is done, you can use parse_test_res.py to automatically calculate the average results. 
For example,
```
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/CoCoOp/rn50_ep100
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/CoCoOp/rn50_ep100 --test-log
```
You can find more evaluation details in [CoOp](https://github.com/KaiyangZhou/CoOp).
## Few-shot learning
The script scripts/ClipFit/main.sh is for few-shot learning. Similar to the base2new setting, please revise the hyper-parameters and then run the experiments. lammbda=8 for dtd and SUN397, and lambda=2 for other datasets. epoch=10 for Food-101 and ImageNet, and epoch=100 for all other dataaset.
```
bash main.sh 
```
You can also use parse_test_res.py to automatically calculate the average results. 


If you use this code in your research, please kindly cite our paper
```
@article{li2024vision,
  title={Vision-language model fine-tuning via simple parameter-efficient modification},
  author={Li, Ming and Zhong, Jike and Li, Chenxin and Li, Liuzhuozheng and Lin, Nie and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:2409.16718},
  year={2024}
}
```
