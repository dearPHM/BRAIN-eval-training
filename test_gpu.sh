mkdir results

python src/baseline_main.py --model=cnn --dataset=cifar --epochs=200 --verbose=0 --local_bs=50 --gpu=0

python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=0 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0
python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=10 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0

python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0

python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=1000 --num_users=21 --byzantines=0 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
