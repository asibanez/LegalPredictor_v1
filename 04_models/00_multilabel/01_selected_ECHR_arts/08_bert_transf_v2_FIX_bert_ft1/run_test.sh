#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/02_preprocessed/00_multilabel/01_selected_ECHR_arts/00_filtered
WORK_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/03_runs/00_multilabel/01_selected_ECHR_arts/01_BERT_TRANSF_v2_FIX_50par_6ep

python test.py \
    --input_dir=$INPUT_DIR \
    --work_dir=$WORK_DIR \
    --test_file=model_dev.pkl \
    --batch_size=1000 \
    --seq_len=256 \
    --num_labels=15 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=50 \
    --pad_idx=0 \
    --gpu_id=0

#read -p 'EOF'

