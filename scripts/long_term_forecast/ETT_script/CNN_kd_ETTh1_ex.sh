export CUDA_VISIBLE_DEVICES=0

model_name=CNN

teacher_name=iTransformer
kd_method=features

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 128 \
  --output_attention \
  --kd_loss_ratio 0.1 \
  --kd \
  --kd_method $kd_method \
  --teacher_path ./checkpoints/long_term_forecast_ETTh1_96_96_iTransformer_ETTh1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df128_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --teacher_model $teacher_name 