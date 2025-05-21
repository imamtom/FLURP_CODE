import os

defense_methods = ['FedAvg', 'FLUD']
attack_methods = ['LabelFlipping', 'SignFlipping', 'Noise', 'ALIE', 'MinMax', 'IPM-01', 'IPM-100', 'Backdoor', 'NoAttack', 'Adaptive']
# # python main.py --dataset AGNews --attack_method Backdoor --defense_method FedAvg --num_clients 20 --iid False --local_learning_rate 0.1 --global_rounds 30 --batch_size 128 --local_epochs 1 --alpha 0.1 --given_size 4096 --gpu_id 0
# 根据攻击手段和防御手段生成对应的 bash 文件
alphas = [0.1, 1, 10]
bash_sentences = []
bash_sentence = 'python main.py --dataset AGNews --attack_method {} --defense_method {} --num_clients 20 --iid False --local_learning_rate 0.1 --global_rounds 30 --batch_size 128 --local_epochs 1 --alpha {} --given_size 4096 --gpu_id 0'
for attack_method in attack_methods:
    for defense_method in defense_methods:
        for alpha in alphas:
            bash_sentences.append(bash_sentence.format(attack_method, defense_method, alpha))

with open('agnews_noniid_bashes.sh', 'w') as f:
    for bash_sentence in bash_sentences:
        f.write(bash_sentence + '\n')
    f.close()


