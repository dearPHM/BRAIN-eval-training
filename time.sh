# rm -rf results
mkdir results

for i in $(seq 1 100); do
    python src/brain_main.py --iid=1 --epochs=50 --frac=0.1 --stale=4 --diff=0.0 --window=4 --threshold=0.0 --verbose=0
done
