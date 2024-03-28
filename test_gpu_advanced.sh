# Baseline - SGD
# python src/baseline_main.py --model=cnn --dataset=cifar --epochs=200 --verbose=0 --local_bs=50 --gpu=0

# Baseline - FedAvg
# python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0

# FedAvg - Byzantine
python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0
# python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0
python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0
python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --frac=0.1 --verbose=0 --local_bs=50 --gpu=0

# FedAsync
# python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0

# FedAsync - Byzantine
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
# python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0

# FedAsync - Stale
# python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=4 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=8 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=16 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=32 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --frac=0.1 --stale=64 --alpha=0.6 --verbose=0 --local_bs=50 --gpu=0

# BRAIN
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0

# BRAIN - Byzantine
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0

# BRAIN - Score Byzantine
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0

# BRAIN - Byzantine & Score Byzantine
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --score_byzantines=11 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=11 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=15 --score_byzantines=15 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0

# BRAIN - Stale
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=8 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=16 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=32 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=64 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0

# # BRAIN - Window (compare w/ Stale)
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=2 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=8 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=16 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=0 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=32 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0

# # BRAIN - Quorum vs Byzantine (diff)
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.25 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.50 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=0.75 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=5 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.25 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.50 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=0.75 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=5 --score_byzantines=10 --frac=0.1 --stale=4 --diff=1.0 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0

# BRAIN - Threshold
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.0 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.05 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.1 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.12 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.125 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.13 --verbose=0 --local_bs=50 --gpu=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.2 --verbose=0 --local_bs=50 --gpu=0
# python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=400 --num_users=21 --byzantines=10 --score_byzantines=0 --frac=0.1 --stale=4 --diff=0.55 --window=4 --threshold=0.3 --verbose=0 --local_bs=50 --gpu=0
