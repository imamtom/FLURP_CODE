## FLURP

### Enviroment

```
conda create -n ENV_NAME python=3.10 -y

conda activate ENV_NAME

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install scikit-learn

pip install matplotlib

pip install pandas

pip install transformers
```

### How to config

- Download the datasets
    - ImageNet-12: Please download the ImageNet-12 subset with this link: [Baidu driver](https://pan.baidu.com/share/init?surl=LjE6g1cxQ98RZHMWi0tQlA) (pwd: qetk) or [Google driver](https://drive.google.com/file/d/1yG9ENDUbOIUKY1i5ADu4X_7Lhbvqca2w/view?usp=sharing) . Please cite [ImageNet-12](https://github.com/bboylyg/RNP) dataset by
    
        ```
        @inproceedings{
        li2023reconstructive,
        title={Reconstructive Neuron Pruning for Backdoor Defense},
        author={Yige Li and Xixiang Lyu and Xingjun Ma and Nodens Koren and Lingjuan Lyu and Bo Li and Yu-Gang Jiang},
        booktitle={ICML},
        year={2023},
        }
        ```

    - AGNews: [HUGGING_FACE](https://huggingface.co/datasets/fancyzhx/ag_news/tree/refs%2Fconvert%2Fparquet/default)



- Config the `my_dir.py` 
    - Config the directories of `datasets_dir` and `results_dir`

### Run

- Run the `main.py` with args: 
    ```bash
    usage: main.py [-h] [--attack_method ATTACK_METHOD] [--defense_method DEFENSE_METHOD] [--num_clients NUM_CLIENTS] [--poisoned_client_portion POISONED_CLIENT_PORTION] [--poison_data_portion POISON_DATA_PORTION] [--dataset_name DATASET_NAME] [--gpu_id GPU_ID] [--iid IID] [--alpha ALPHA] [--global_rounds GLOBAL_ROUNDS] [--local_epochs LOCAL_EPOCHS] [--local_learning_rate LOCAL_LEARNING_RATE] [--local_momentum LOCAL_MOMENTUM] [--batch_size BATCH_SIZE] [--given_size GIVEN_SIZE] [--target_label TARGET_LABEL]
    ``` 
Examples of bash: `train_bash_example.sh`