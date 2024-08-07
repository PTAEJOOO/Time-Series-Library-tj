export CUDA_VISIBLE_DEVICES=0

model_name=CNN
teacher_name=iTransformer
kd_loss_ratio="0.1 0.3 0.5 0.7"
kd_method=features

for kd in $kd_loss_ratio
do
  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Monthly' \
    --model_id m4_Monthly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Monthly_iTransformer_m4_ftM_sl36_ll18_pl18_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Yearly' \
    --model_id m4_Yearly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Yearly_iTransformer_m4_ftM_sl12_ll6_pl6_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Quarterly' \
    --model_id m4_Quarterly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Quarterly_iTransformer_m4_ftM_sl16_ll8_pl8_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Weekly' \
    --model_id m4_Weekly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Weekly_iTransformer_m4_ftM_sl26_ll13_pl13_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Daily' \
    --model_id m4_Daily \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Daily_iTransformer_m4_ftM_sl28_ll14_pl14_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name

  python -u run.py \
    --task_name short_term_forecast \
    --is_training 1 \
    --root_path ./dataset/m4 \
    --seasonal_patterns 'Hourly' \
    --model_id m4_Hourly \
    --model $model_name \
    --data m4 \
    --features M \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'SMAPE' \
    --output_attention \
    --kd_loss_ratio $kd \
    --kd \
    --kd_method $kd_method \
    --teacher_path ./checkpoints/short_term_forecast_m4_Hourly_iTransformer_m4_ftM_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
    --teacher_model $teacher_name
done