mkdir -p ./save
mkdir -p ./trainlogs

export PYTHONPATH=$PYTHONPATH:YOUR_PYTHON_PATH
export CUDA_VISIBLE_DEVICES=0

method=ntkmtl
seed=0
ntk_exp=0.75

python -u trainer.py \
 --method=$method \
 --seed=$seed  \
 --ntk_exp=$ntk_exp \
 --wandb_logger_name "XXX" \
 --wandb_project=XXX \
 --wandb_entity=XXX