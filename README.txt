
Data preparation:
    Download the corresponding dataset in the folder './data/', an example of file list for clothing1M has been presented.



Train and test:
    To run the proposed ULC approach on CIFAR-10 with 50% symmetric noise under 1:10 imbalance, execute 'python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --r 0.5 --imbalance 0.1 --noise_mode sym'

    To run the proposed ULC approach on CIFAR-100 with 50% symmetric noise under 1:10 imbalance, execute 'python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --r 0.5 --imbalance 0.1 --noise_mode sym --lambda_u 150'

    To run the proposed ULC approach on clothing1M, execute 'python Train_clothing1M.py'
