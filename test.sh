python src/baseline_main.py --model=cnn --dataset=cifar --epochs=50 --verbose=0

python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=0 --frac=0.1 --verbose=0
python src/federated_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=10 --frac=0.1 --verbose=0

python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=0 --frac=0.1 --stale=4 --verbose=0
python src/fedAsync_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=10 --frac=0.1 --stale=4 --verbose=0

python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=0 --score_byzantines=0 --frac=0.1 --stale=4 --window=4 --verbose=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=10 --score_byzantines=0 --frac=0.1 --stale=4 --window=4 --verbose=0
python src/brain_main.py --model=cnn --dataset=cifar --iid=1 --epochs=50 --num_users=21 --byzantine=0 --score_byzantines=10 --frac=0.1 --stale=4 --window=4 --verbose=0
