python train.py --dataset FrcSub --cdm facd --device cuda:0 --lr 3e-3 --batch_size 128 --num_epochs 1 --decoder irt --seed 0 --test_size 0.8
python test.py --dataset FrcSub --strategy BECAT --cdm facd --device cuda:0 --lr 3e-3 --batch_size 128 --num_epochs 2 --decoder irt --seed 0 --test_size 0.8 --test_length 15

python train.py --dataset FrcSub --cdm facd --device cuda:0 --lr 3e-3 --batch_size 128 --num_epochs 1 --decoder ncd --seed 0 --test_size 0.8
python test.py --dataset FrcSub --strategy BECAT --cdm facd --device cuda:0 --lr 3e-3 --batch_size 128 --num_epochs 2 --decoder ncd --seed 0 --test_size 0.8 --test_length 15