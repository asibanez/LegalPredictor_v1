#INPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/02_full_binary
#OUTPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/02_runs/01_binary/00_BERT_TRANSF_v2_FIX_50par_20ep/

INPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/01_preprocessed/01_50pars_256_tok/04_full_binary_resampled_1.0
OUTPUT_DIR=/data/rsg/nlp/sibanez/02_LegalOutcomePredictor/00_data/v2/02_runs/01_binary/02_BERT_TRANSF_v2_FIX_50par_10ep_resampled_1.0

python train_v1.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --n_epochs=20 \
    --batch_size=40 \
    --shuffle_train=True \
    --train_toy_data=False \
    --len_train_toy_data=800 \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --seed=1234 \
    --seq_len=256 \
    --num_labels=1 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars_facts=50 \
    --max_n_pars_echr=6 \
    --pad_idx=0 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --use_cuda=True \
    --gpu_ids=0,1,2,3,4,5,6,7

#read -p 'EOF'

#--batch_size=50
#--n_epochs=100
#--max_n_pars=200
