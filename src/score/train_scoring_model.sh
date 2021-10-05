MODEL_DIR=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/Fragment_stitching/src/models/score_intermediate
WEIGHTS_SAVE_PATH=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/runs
NAME='balanced_lig_clash_score'
DATA_DIR=/oak/stanford/groups/rondror/projects/combind/flexibility/atom3d/ml_score/balanced_lmdb_data/split/data
cd $MODEL_DIR
export DATA_DIR=$DATA_DIR
export MODEL_DIR=$MODEL_DIR
export WEIGHTS_SAVE_PATH=$WEIGHTS_SAVE_PATH
TASK='regression'
echo $WEIGHTS_SAVE_PATH
echo $MODEL_DIR

python train.py -train $DATA_DIR/train -val $DATA_DIR/val -test $DATA_DIR/test --weights_save_path $WEIGHTS_SAVE_PATH --gpus=1 --num_workers=8 --batch_size=8 --accumulate_grad_batches=2 --learning_rate=0.01 --max_epochs=40 --project_name pdbbind --run_id balanced_lig_clash_score --balance_dataset