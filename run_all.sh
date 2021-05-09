seed="${1:-0}" 
device="${2:-cuda:0}"
data="${3:-/data/datasets}"
echo "Using seed" $seed
echo "Using device" $device
echo "Using dataset location" $data

python nonlifelong.py --seed $seed --device $device --data-root $data --model FGN   --optim SGD
python    lifelong.py --seed $seed --device $device --data-root $data --model FGN   --optim SGD

python nonlifelong.py --seed $seed --device $device --data-root $data --model GAT   --optim Adam
python    lifelong.py --seed $seed --device $device --data-root $data --model GAT   --optim Adam

python nonlifelong.py --seed $seed --device $device --data-root $data --model AFGN  --optim Adam
python    lifelong.py --seed $seed --device $device --data-root $data --model AFGN  --optim Adam

python nonlifelong.py --seed $seed --device $device --data-root $data --model GCN   --optim SGD
python    lifelong.py --seed $seed --device $device --data-root $data --model GCN   --optim SGD

python nonlifelong.py --seed $seed --device $device --data-root $data --model APPNP --optim SGD
python    lifelong.py --seed $seed --device $device --data-root $data --model APPNP --optim SGD
