# HSP
Now debugging

```
set DATASET_ROOT=
python train_hsp.py -b 3 --epoch 100 --gpu 0 --dataset %DATASET_ROOT%\volume-drr-v9 --train-index %DATASET_ROOT%\k-fold-index\4-fold\4_0_train.txt --valid-index %DATASET_ROOT%\k-fold-index\4-fold\4_0_valid.txt --debug --n-level 4
```
