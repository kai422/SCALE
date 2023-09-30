Our Code is based on [OpenOOD: Benchmarking Generalized OOD Detection](https://github.com/Jingkang50/OpenOOD):

# Environment and dataset 
pip install -e .

Prepare Dataset and pretrained network following [OpenOOD](https://github.com/Jingkang50/OpenOOD) official instruction.
You need to prepare following dataset:
```
python ./scripts/download.py \
	--contents 'datasets' 'checkpoints' \
	--save_dir './data' './results' \
	--dataset_mode 'benchmark'
```
# SCALE as post hoc model enhancement.

```
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor scale \
    --save-score --save-csv

```

# IS as training time model enhancement.
perform inference on ISH model:
```
python scripts/eval_ood_imagenet.py \
  --ckpt-path results/ish/last.ckpt \
  --arch resnet50 \
  --postprocessor scale \
  --save-score --save-csv

```