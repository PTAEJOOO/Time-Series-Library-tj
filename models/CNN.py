import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

class Model(nn.Module):
    """
    1d-CNN for time-series forecasting
    """
    def __init__(self, configs, individual=False):

        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.individual = individual
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.data = configs.data
        self.d_model = configs.d_model

        self.kd = configs.kd
        self.kd_method = configs.kd_method

        self.batch_size = configs.batch_size
        self.n_heads = configs.n_heads

        if self.data == 'm4':
            self.conv1 = nn.Conv1d(in_channels=self.channels, out_channels=16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * self.seq_len, 64)
            self.fc2 = nn.Linear(64, self.pred_len)
            self.relu = nn.ReLU()
        else:
            ## Small CNN
            self.conv1 = nn.Conv1d(in_channels=self.channels, out_channels=16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * self.seq_len, 128)
            self.fc2 = nn.Linear(128, self.pred_len * self.c_out)
            self.relu = nn.ReLU()
            
            ## Large CNN
            # self.conv1 = nn.Conv1d(in_channels=self.channels, out_channels=16, kernel_size=3, padding=1)
            # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            # self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

            # self.fc1 = nn.Linear(128 * self.seq_len, 128)
            # self.fc2 = nn.Linear(128, self.pred_len * self.c_out)
            # self.relu = nn.ReLU()
        
        if self.kd_method == 'features': # & teacher_model == 'iTransformer'
            if self.data == 'm4':
                self.q_conv = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
                self.k_conv = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
                self.v_conv = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

                self.q_fc = nn.Linear(self.seq_len, self.d_model)
                self.k_fc = nn.Linear(self.seq_len, self.d_model)
                self.v_fc = nn.Linear(self.seq_len, self.d_model)
            else:
                self.q_conv = nn.Conv1d(in_channels=128, out_channels=11, kernel_size=3, padding=1)
                self.k_conv = nn.Conv1d(in_channels=128, out_channels=11, kernel_size=3, padding=1)
                self.v_conv = nn.Conv1d(in_channels=128, out_channels=11, kernel_size=3, padding=1)

                self.q_fc = nn.Linear(self.seq_len, self.d_model)
                self.k_fc = nn.Linear(self.seq_len, self.d_model)
                self.v_fc = nn.Linear(self.seq_len, self.d_model)

    def encoder(self, x):
        if self.data == 'm4':
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            conv_out = nn.functional.relu(x)

            x = conv_out.view(conv_out.size(0), -1)  # Flatten the tensor
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

            x = x.unsqueeze(2)

        else:
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            ## Large CNN
            # x = nn.functional.relu(x)
            # x = self.conv3(x)
            # x = nn.functional.relu(x)
            # x = self.conv4(x)
            ##
            conv_out = nn.functional.relu(x)

            x = conv_out.view(conv_out.size(0), -1)  # Flatten the tensor
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

            x = x.view(x.size(0), -1, self.c_out)

        if self.kd_method == 'features':
            return x, conv_out
        else:
            return x
    
    def forecast(self, x_enc):
        return self.encoder(x_enc)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.kd_method == 'features':
                dec_out, conv_out = self.forecast(x_enc)

                query = self.q_fc(self.q_conv(conv_out))
                key = self.k_fc(self.k_conv(conv_out))
                value = self.v_fc(self.v_conv(conv_out))

                B,L,_ = query.shape
                query = query.view(B,L,self.n_heads,-1)
                key = key.view(B,L,self.n_heads,-1)
                value = value.view(B,L,self.n_heads,-1)

                return dec_out[:, -self.pred_len:, :], query, key, value
            else:
                dec_out = self.forecast(x_enc)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        return None