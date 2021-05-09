seed="${1:-0}" 
device="${2:-cuda:0}"
echo "Using seed" $seed
echo "Using device" $device

python nonlifelong.py --seed $seed --device $device --model FGN   --optim SGD 
python    lifelong.py --seed $seed --device $device --model FGN   --optim SGD

python nonlifelong.py --seed $seed --device $device --model GAT   --optim Adam
python    lifelong.py --seed $seed --device $device --model GAT   --optim Adam

python nonlifelong.py --seed $seed --device $device --model AFGN  --optim Adam
python    lifelong.py --seed $seed --device $device --model AFGN  --optim Adam

python nonlifelong.py --seed $seed --device $device --model GCN   --optim SGD
python    lifelong.py --seed $seed --device $device --model GCN   --optim SGD

python nonlifelong.py --seed $seed --device $device --model APPNP --optim SGD 
python    lifelong.py --seed $seed --device $device --model APPNP --optim SGD
