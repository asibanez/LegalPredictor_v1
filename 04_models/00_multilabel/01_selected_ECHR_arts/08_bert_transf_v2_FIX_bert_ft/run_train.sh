#INPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\01_preprocessed\\03_toy_3
#OUTPUT_DIR=C:\\Users\\siban\\Dropbox\\CSAIL\\Projects\\12_Legal_Outcome_Predictor\\00_data\\v2\\02_runs\\00_TEST_DELETE

INPUT_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/02_preprocessed/00_multilabel/01_selected_ECHR_arts/00_filtered
OUTPUT_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/03_runs/00_multilabel/01_selected_ECHR_arts/01_BERT_TRANSF_v2_FIX_50par_6ep

python train_###.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --n_epochs=6 \
    --batch_size=14 \
    --shuffle_train=True \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=256 \
    --num_labels=15 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars=50 \
    --pad_idx=0 \
    --save_final_model=True \
    --save_model_steps=True \
    --use_cuda=True \
    --gpu_ids=0,1

#read -p 'EOF'

#--batch_size=6000
#--n_epochs=100
#--max_n_pars=200
