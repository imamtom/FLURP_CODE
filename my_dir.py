# used to store the directory of the datasets and the results
# for the FLURP project

import os

datasets_dir = {'MNIST': '/home/lwj/MNIST',
                'FashionMNIST':'/home/lwj/FashionMNIST',
                'CIFAR10': '/home/lwj/CIFAR10',

                # /home/lwj/AGNews/test.csv
                # /home/lwj/AGNews/train.csv
                'AGNews':'/home/lwj/AGNews',  
                
                # /home/lwj/ImageNet12/train
                # /home/lwj/ImageNet12/val
                'ImageNet12':'/home/lwj/ImageNet12',
}


results_dir = {'MNIST': '/home/lwj/FLURP_results/MNIST',
                'FashionMNIST':'/home/lwj/FLURP_results/FashionMNIST',
                'CIFAR10': '/home/lwj/FLURP_results/CIFAR10',

                'AGNews':'/home/lwj/FLURP_results/AGNews',
                'ImageNet12':'/home/lwj/FLURP_results/ImageNet12',
}

