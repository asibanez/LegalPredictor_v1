#INPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/01_preprocessed/01_50pars_256_tok/02_full_binary
#OUTPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/12_Legal_Outcome_Predictor/00_data/v2/02_runs/01_binary/00_BERT_TRANSF_v2_FIX_50par_20ep/

INPUT_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/02_preprocessed/01_binary/01_selected_ECHR_arts/00_filtered
OUTPUT_DIR=/data/rsg/nlp/sibanez/03_LegalPredictor_v1/00_data/03_runs/01_binary/01_selected_ECHR_arts/01_TEST_DELETE/02_BERTx2_TRANSF_v2_FIX_50par_10ep_improved

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --seq_len=256 \
    --num_labels=1 \
    --n_heads=8 \
    --hidden_dim=512 \
    --max_n_pars_facts=50 \
    --max_n_pars_echr=6 \
    --pad_idx=0 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=50 \
    --shuffle_train=True \
    --drop_last_train=True \
    --dev_train_ratio=2 \
    --train_toy_data=False \
    --len_train_toy_data=30 \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0,1,2,3,4,5,6,7 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.1 \
    --batch_size_test=4 \
    --gpu_id_test=0 \

#read -p 'EOF'

#--task=Train / Test
#--batch_size=40
#--batch_size=25 / 0,1,2,3
#--n_epochs=100
#--max_n_pars=200
