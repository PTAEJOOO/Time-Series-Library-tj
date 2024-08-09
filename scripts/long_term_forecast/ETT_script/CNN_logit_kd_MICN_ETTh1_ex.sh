export CUDA_VISIBLE_DEVICES=0

model_name=CNN

teacher_name=MICN
kd_loss_ratio="0.1"
kd_method=logits

for kd in $kd_loss_ratio
do
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --kd_loss_ratio $kd \
  --kd \
  --kd_method $kd_method \
  --teacher_path ./checkpoints/long_term_forecast_ETTh1_96_96_MICN_ETTh1_ftM_sl96_ll96_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --teacher_model $teacher_name 

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_192 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/long_term_forecast_ETTh1_96_192_MICN_ETTh1_ftM_sl96_ll96_pl192_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_336 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/long_term_forecast_ETTh1_96_336_MICN_ETTh1_ftM_sl96_ll96_pl336_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_720 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/long_term_forecast_ETTh1_96_720_MICN_ETTh1_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name
done