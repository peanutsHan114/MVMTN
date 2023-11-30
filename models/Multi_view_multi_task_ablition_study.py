
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MyEmbeddingLayer(nn.Module):
    def __init__(self,embsize_list,num_list) -> None:
        super().__init__()
        self.act_embedding=nn.Embedding(num_list[0]+1,embsize_list[0],padding_idx=num_list[0])
        self.res_embedding=nn.Embedding(num_list[1]+1,embsize_list[1],padding_idx=num_list[1])
        # self.last_embedding=nn.Embedding(num_list[2]+1,embsize_list[2])
        # self.dif_embedding=nn.Embedding(num_list[3]+1,embsize_list[3])
        nn.init.normal_(self.act_embedding.weight, std=0.1)
        nn.init.normal_(self.res_embedding.weight, std=0.1)
        # nn.init.normal_(self.last_embedding.weight, std=0.1)
        # nn.init.normal_(self.dif_embedding.weight, std=0.1)


    def forward(self,data):
        act_embedding=self.act_embedding(data[:,:,0].long())
        res_embedding=self.res_embedding(data[:,:,1].long())
        # last_embedding=self.last_embedding(data[:,:,2].long())
        # dif_embedding=self.dif_embedding(data[:,:,3].long())
        # x=torch.cat((act_embedding,data[:,:,2:]),dim=-1)
        x=torch.cat((act_embedding,res_embedding,data[:,:,2:]),dim=-1)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Multi_view_multi_task(nn.Module):
    def __init__(self,embsize_list,num_list,hidden_dim,kernel_size,dropout,num_layers,pattern_embsize_list,pattern_num_list):
        super().__init__()
        self.embedding = MyEmbeddingLayer(embsize_list,num_list)
        # self.encoder=nn.Linear(sum(embsize_list[0:2])+2,32)

        self.tcn_layer=nn.ModuleList([TemporalConvNet(sum(embsize_list[0:2])+2,[hidden_dim]*num_layers, kernel_size=kernel_size, dropout=dropout) for i in range(3)])
        self.lstm_layer=nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim,batch_first=True) for i in range(3)])
        self.bn_layer=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(3)])

        self.task_aggregation_layer=nn.MultiheadAttention(hidden_dim,4,batch_first=True)

        self.time_decoder=nn.Sequential(
            nn.Linear(hidden_dim,1)
        )

        self.act_pattern_embedding=nn.Embedding(num_list[0]+1,pattern_embsize_list[0],padding_idx=num_list[0])
        self.act_pattern_tcn=TemporalConvNet(embsize_list[0],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_res_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_res_res_pattern_embedding=nn.Embedding(pattern_num_list[1]+2,pattern_embsize_list[1],padding_idx=pattern_num_list[1]+1)
        self.act_res_pattern_tcn=TemporalConvNet(sum(pattern_embsize_list[0:2]),[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_res_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_res_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_time_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_time_time_pattern_embedding=nn.Embedding(pattern_num_list[2]+2,pattern_embsize_list[2],padding_idx=pattern_num_list[2]+1)
        self.act_time_pattern_tcn=TemporalConvNet(pattern_embsize_list[0]+pattern_embsize_list[2],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_time_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_time_pattern_bn=nn.BatchNorm1d(hidden_dim)



        self.act_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.act_decoder=nn.Linear(hidden_dim,num_list[0])

        self.res_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.res_decoder=nn.Linear(hidden_dim,num_list[1])

        self.time_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.time_decoder=nn.Linear(hidden_dim,1)

        self.tanh=nn.Tanh()

    def operate_res(self,act_res_pattern):
        act_embedding=self.act_res_act_pattern_embedding(act_res_pattern[:,:,0].long())
        res_embedding=self.act_res_res_pattern_embedding(act_res_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,res_embedding],dim=-1)
        tcnout=self.act_res_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_res_pattern_lstm(tcnout)[0]
        bn=self.act_res_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn
    
    def operate_time(self,act_time_pattern):
        act_embedding=self.act_time_act_pattern_embedding(act_time_pattern[:,:,0].long())
        time_embedding=self.act_time_time_pattern_embedding(act_time_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,time_embedding],dim=-1)
        tcnout=self.act_time_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_time_pattern_lstm(tcnout)[0]
        bn=self.act_time_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn

    def operate_act(self,act_pattern):
        pattern_embedding=self.act_pattern_embedding(act_pattern.long())
        pattern_tcn=self.act_pattern_tcn(pattern_embedding.transpose(1,2)).transpose(1,2)
        pattern_lstm=self.act_pattern_lstm(pattern_tcn)[0]
        bn_output=self.act_pattern_bn(pattern_lstm.contiguous().view(-1,pattern_lstm.shape[2])).view(pattern_lstm.shape)
        return bn_output

    def forward(self,data,act_pattern,act_res_pattern,act_time_pattern):
        embedding=self.embedding(data)
        # encoder=self.encoder(embedding)
        tcnout=[]
        for i in range(3):
            tcnout.append(self.tcn_layer[i](embedding.transpose(1, 2)).transpose(1, 2))
        lstmout=[]
        for i in range(3):
            # lstmout.append(self.lstm_layer[i](tcnout[i])[0])
            lstmout1=self.lstm_layer[i](tcnout[i])[0]
            # lstmout1=lstmout1.permute(0, 2, 1).contiguous()
            bn1=self.bn_layer[i](lstmout1.contiguous().view(-1,lstmout1.shape[2])).view(lstmout1.shape)
            # bn1=bn1.permute(0, 2, 1).contiguous()
            # lstmout.append(bn1[:,-1,:].unsqueeze(dim=1))
            lstmout.append(bn1)
            # lstmout.append(lstmout1[:,-1,:].unsqueeze(dim=1))

        avgout=[]
        for i in range(3):
            avgout.append(torch.mean(lstmout[i],axis=1).unsqueeze(dim=1))
        lstmagg=torch.cat(avgout,dim=1)
        # lstmagg=torch.cat(lstmout,dim=1)
        task_aggreation,weights=self.task_aggregation_layer(lstmagg,lstmagg,lstmagg)
        
        A,B,C=lstmout[0],lstmout[1],lstmout[2]
        W=weights
        A_prime = (A * W[:, 0, 0].view(-1, 1, 1) + B * W[:, 0, 1].view(-1, 1, 1) + C * W[:, 0, 2].view(-1, 1, 1))
        B_prime = (A * W[:, 1, 0].view(-1, 1, 1) + B * W[:, 1, 1].view(-1, 1, 1) + C * W[:, 1, 2].view(-1, 1, 1))
        C_prime = (A * W[:, 2, 0].view(-1, 1, 1) + B * W[:, 2, 1].view(-1, 1, 1) + C * W[:, 2, 2].view(-1, 1, 1))


        act_pattern=self.operate_act(act_pattern)
        act_res_pattern=self.operate_res(act_res_pattern)
        act_time_pattern=self.operate_time(act_time_pattern)
        
        act_attn,weights=self.act_attention(A_prime,act_pattern,act_pattern)
        res_attn,weights=self.res_attention(B_prime,act_res_pattern,act_res_pattern)
        time_attn,weights=self.time_attention(C_prime,act_time_pattern,act_time_pattern)

        # lsact=torch.cat((act_pattern.unsqueeze(dim=1),task_aggreation[:,0,:].unsqueeze(dim=1)),dim=1)
        # lsact_res=torch.cat((act_res_pattern.unsqueeze(dim=1),task_aggreation[:,1,:].unsqueeze(dim=1)),dim=1)
        # lsact_time=torch.cat((act_time_pattern.unsqueeze(dim=1),task_aggreation[:,2,:].unsqueeze(dim=1)),dim=1)
        # act_attn,weights=self.act_attention(lsact,lsact,lsact)
        # res_attn,weights=self.res_attention(lsact_res,lsact_res,lsact_res)
        # time_attn,weights=self.time_attention(lsact_time,lsact_time,lsact_time)

        # act_output=self.act_decoder(act_attn[:,-1,:])

        # act_out=self.act_decoder(act_attn[:,1,:])
        # res_out=self.res_decoder(res_attn[:,1,:])
        # time_out=self.time_decoder(time_attn[:,1,:])

        act_out=self.act_decoder(act_attn[:,-1,:])
        res_out=self.res_decoder(res_attn[:,-1,:])
        time_out=self.time_decoder(time_attn[:,-1,:])


        return act_out,res_out,time_out



class Multi_view_multi_task_no_frequent(nn.Module):
    def __init__(self,embsize_list,num_list,hidden_dim,kernel_size,dropout,num_layers,pattern_embsize_list,pattern_num_list):
        super().__init__()
        self.embedding = MyEmbeddingLayer(embsize_list,num_list)
        # self.encoder=nn.Linear(sum(embsize_list[0:2])+2,32)

        self.tcn_layer=nn.ModuleList([TemporalConvNet(sum(embsize_list[0:2])+2,[hidden_dim]*num_layers, kernel_size=kernel_size, dropout=dropout) for i in range(3)])
        self.lstm_layer=nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim,batch_first=True) for i in range(3)])
        self.bn_layer=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(3)])

        self.task_aggregation_layer=nn.MultiheadAttention(hidden_dim,4,batch_first=True)

        self.time_decoder=nn.Sequential(
            nn.Linear(hidden_dim,1)
        )

        self.act_pattern_embedding=nn.Embedding(num_list[0]+1,pattern_embsize_list[0],padding_idx=num_list[0])
        self.act_pattern_tcn=TemporalConvNet(embsize_list[0],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_res_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_res_res_pattern_embedding=nn.Embedding(pattern_num_list[1]+2,pattern_embsize_list[1],padding_idx=pattern_num_list[1]+1)
        self.act_res_pattern_tcn=TemporalConvNet(sum(pattern_embsize_list[0:2]),[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_res_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_res_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_time_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_time_time_pattern_embedding=nn.Embedding(pattern_num_list[2]+2,pattern_embsize_list[2],padding_idx=pattern_num_list[2]+1)
        self.act_time_pattern_tcn=TemporalConvNet(pattern_embsize_list[0]+pattern_embsize_list[2],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_time_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_time_pattern_bn=nn.BatchNorm1d(hidden_dim)



        self.act_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.act_decoder=nn.Linear(hidden_dim,num_list[0])

        self.res_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.res_decoder=nn.Linear(hidden_dim,num_list[1])

        self.time_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.time_decoder=nn.Linear(hidden_dim,1)

        self.tanh=nn.Tanh()

    def operate_res(self,act_res_pattern):
        act_embedding=self.act_res_act_pattern_embedding(act_res_pattern[:,:,0].long())
        res_embedding=self.act_res_res_pattern_embedding(act_res_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,res_embedding],dim=-1)
        tcnout=self.act_res_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_res_pattern_lstm(tcnout)[0]
        bn=self.act_res_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn
    
    def operate_time(self,act_time_pattern):
        act_embedding=self.act_time_act_pattern_embedding(act_time_pattern[:,:,0].long())
        time_embedding=self.act_time_time_pattern_embedding(act_time_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,time_embedding],dim=-1)
        tcnout=self.act_time_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_time_pattern_lstm(tcnout)[0]
        bn=self.act_time_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn

    def operate_act(self,act_pattern):
        pattern_embedding=self.act_pattern_embedding(act_pattern.long())
        pattern_tcn=self.act_pattern_tcn(pattern_embedding.transpose(1,2)).transpose(1,2)
        pattern_lstm=self.act_pattern_lstm(pattern_tcn)[0]
        bn_output=self.act_pattern_bn(pattern_lstm.contiguous().view(-1,pattern_lstm.shape[2])).view(pattern_lstm.shape)
        return bn_output

    def forward(self,data,act_pattern,act_res_pattern,act_time_pattern):
        embedding=self.embedding(data)
        # encoder=self.encoder(embedding)
        tcnout=[]
        for i in range(3):
            tcnout.append(self.tcn_layer[i](embedding.transpose(1, 2)).transpose(1, 2))
        lstmout=[]
        for i in range(3):
            # lstmout.append(self.lstm_layer[i](tcnout[i])[0])
            lstmout1=self.lstm_layer[i](tcnout[i])[0]
            # lstmout1=lstmout1.permute(0, 2, 1).contiguous()
            bn1=self.bn_layer[i](lstmout1.contiguous().view(-1,lstmout1.shape[2])).view(lstmout1.shape)
            # bn1=bn1.permute(0, 2, 1).contiguous()
            # lstmout.append(bn1[:,-1,:].unsqueeze(dim=1))
            lstmout.append(bn1)
            # lstmout.append(lstmout1[:,-1,:].unsqueeze(dim=1))

        avgout=[]
        for i in range(3):
            avgout.append(torch.mean(lstmout[i],axis=1).unsqueeze(dim=1))
        lstmagg=torch.cat(avgout,dim=1)
        # lstmagg=torch.cat(lstmout,dim=1)
        task_aggreation,weights=self.task_aggregation_layer(lstmagg,lstmagg,lstmagg)
        
        A,B,C=lstmout[0],lstmout[1],lstmout[2]
        W=weights
        A_prime = (A * W[:, 0, 0].view(-1, 1, 1) + B * W[:, 0, 1].view(-1, 1, 1) + C * W[:, 0, 2].view(-1, 1, 1))
        B_prime = (A * W[:, 1, 0].view(-1, 1, 1) + B * W[:, 1, 1].view(-1, 1, 1) + C * W[:, 1, 2].view(-1, 1, 1))
        C_prime = (A * W[:, 2, 0].view(-1, 1, 1) + B * W[:, 2, 1].view(-1, 1, 1) + C * W[:, 2, 2].view(-1, 1, 1))


        # act_pattern=self.operate_act(act_pattern)
        # act_res_pattern=self.operate_res(act_res_pattern)
        # act_time_pattern=self.operate_time(act_time_pattern)
        
        # act_attn,weights=self.act_attention(A_prime,act_pattern,act_pattern)
        # res_attn,weights=self.res_attention(B_prime,act_res_pattern,act_res_pattern)
        # time_attn,weights=self.time_attention(C_prime,act_time_pattern,act_time_pattern)

        # lsact=torch.cat((act_pattern.unsqueeze(dim=1),task_aggreation[:,0,:].unsqueeze(dim=1)),dim=1)
        # lsact_res=torch.cat((act_res_pattern.unsqueeze(dim=1),task_aggreation[:,1,:].unsqueeze(dim=1)),dim=1)
        # lsact_time=torch.cat((act_time_pattern.unsqueeze(dim=1),task_aggreation[:,2,:].unsqueeze(dim=1)),dim=1)
        # act_attn,weights=self.act_attention(lsact,lsact,lsact)
        # res_attn,weights=self.res_attention(lsact_res,lsact_res,lsact_res)
        # time_attn,weights=self.time_attention(lsact_time,lsact_time,lsact_time)

        # act_output=self.act_decoder(act_attn[:,-1,:])

        # act_out=self.act_decoder(act_attn[:,1,:])
        # res_out=self.res_decoder(res_attn[:,1,:])
        # time_out=self.time_decoder(time_attn[:,1,:])

        # act_out=self.act_decoder(act_attn[:,-1,:])
        # res_out=self.res_decoder(res_attn[:,-1,:])
        # time_out=self.time_decoder(time_attn[:,-1,:])


        act_out=self.act_decoder(A_prime[:,-1,:])
        res_out=self.res_decoder(B_prime[:,-1,:])
        time_out=self.time_decoder(C_prime[:,-1,:])



        return act_out,res_out,time_out


class Multi_view_multi_task_no_task_relevance(nn.Module):
    def __init__(self,embsize_list,num_list,hidden_dim,kernel_size,dropout,num_layers,pattern_embsize_list,pattern_num_list):
        super().__init__()
        self.embedding = MyEmbeddingLayer(embsize_list,num_list)
        # self.encoder=nn.Linear(sum(embsize_list[0:2])+2,32)

        self.tcn_layer=nn.ModuleList([TemporalConvNet(sum(embsize_list[0:2])+2,[hidden_dim]*num_layers, kernel_size=kernel_size, dropout=dropout) for i in range(3)])
        self.lstm_layer=nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim,batch_first=True) for i in range(3)])
        self.bn_layer=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(3)])

        self.task_aggregation_layer=nn.MultiheadAttention(hidden_dim,4,batch_first=True)

        self.time_decoder=nn.Sequential(
            nn.Linear(hidden_dim,1)
        )

        self.act_pattern_embedding=nn.Embedding(num_list[0]+1,pattern_embsize_list[0],padding_idx=num_list[0])
        self.act_pattern_tcn=TemporalConvNet(embsize_list[0],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_res_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_res_res_pattern_embedding=nn.Embedding(pattern_num_list[1]+2,pattern_embsize_list[1],padding_idx=pattern_num_list[1]+1)
        self.act_res_pattern_tcn=TemporalConvNet(sum(pattern_embsize_list[0:2]),[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_res_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_res_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_time_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_time_time_pattern_embedding=nn.Embedding(pattern_num_list[2]+2,pattern_embsize_list[2],padding_idx=pattern_num_list[2]+1)
        self.act_time_pattern_tcn=TemporalConvNet(pattern_embsize_list[0]+pattern_embsize_list[2],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_time_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_time_pattern_bn=nn.BatchNorm1d(hidden_dim)



        self.act_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.act_decoder=nn.Linear(hidden_dim,num_list[0])

        self.res_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.res_decoder=nn.Linear(hidden_dim,num_list[1])

        self.time_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True)
        self.time_decoder=nn.Linear(hidden_dim,1)

        self.tanh=nn.Tanh()

    def operate_res(self,act_res_pattern):
        act_embedding=self.act_res_act_pattern_embedding(act_res_pattern[:,:,0].long())
        res_embedding=self.act_res_res_pattern_embedding(act_res_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,res_embedding],dim=-1)
        tcnout=self.act_res_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_res_pattern_lstm(tcnout)[0]
        bn=self.act_res_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn
    
    def operate_time(self,act_time_pattern):
        act_embedding=self.act_time_act_pattern_embedding(act_time_pattern[:,:,0].long())
        time_embedding=self.act_time_time_pattern_embedding(act_time_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,time_embedding],dim=-1)
        tcnout=self.act_time_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_time_pattern_lstm(tcnout)[0]
        bn=self.act_time_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn

    def operate_act(self,act_pattern):
        pattern_embedding=self.act_pattern_embedding(act_pattern.long())
        pattern_tcn=self.act_pattern_tcn(pattern_embedding.transpose(1,2)).transpose(1,2)
        pattern_lstm=self.act_pattern_lstm(pattern_tcn)[0]
        bn_output=self.act_pattern_bn(pattern_lstm.contiguous().view(-1,pattern_lstm.shape[2])).view(pattern_lstm.shape)
        return bn_output

    def forward(self,data,act_pattern,act_res_pattern,act_time_pattern):
        embedding=self.embedding(data)
        # encoder=self.encoder(embedding)
        tcnout=[]
        for i in range(3):
            tcnout.append(self.tcn_layer[i](embedding.transpose(1, 2)).transpose(1, 2))
        lstmout=[]
        for i in range(3):
            # lstmout.append(self.lstm_layer[i](tcnout[i])[0])
            lstmout1=self.lstm_layer[i](tcnout[i])[0]
            # lstmout1=lstmout1.permute(0, 2, 1).contiguous()
            bn1=self.bn_layer[i](lstmout1.contiguous().view(-1,lstmout1.shape[2])).view(lstmout1.shape)
            # bn1=bn1.permute(0, 2, 1).contiguous()
            # lstmout.append(bn1[:,-1,:].unsqueeze(dim=1))
            lstmout.append(bn1)
            # lstmout.append(lstmout1[:,-1,:].unsqueeze(dim=1))

        # avgout=[]
        # for i in range(3):
        #     avgout.append(torch.mean(lstmout[i],axis=1).unsqueeze(dim=1))
        # lstmagg=torch.cat(avgout,dim=1)
        # # lstmagg=torch.cat(lstmout,dim=1)
        # task_aggreation,weights=self.task_aggregation_layer(lstmagg,lstmagg,lstmagg)
        
        # A,B,C=lstmout[0],lstmout[1],lstmout[2]
        # W=weights
        # A_prime = (A * W[:, 0, 0].view(-1, 1, 1) + B * W[:, 0, 1].view(-1, 1, 1) + C * W[:, 0, 2].view(-1, 1, 1))
        # B_prime = (A * W[:, 1, 0].view(-1, 1, 1) + B * W[:, 1, 1].view(-1, 1, 1) + C * W[:, 1, 2].view(-1, 1, 1))
        # C_prime = (A * W[:, 2, 0].view(-1, 1, 1) + B * W[:, 2, 1].view(-1, 1, 1) + C * W[:, 2, 2].view(-1, 1, 1))


        act_pattern=self.operate_act(act_pattern)
        act_res_pattern=self.operate_res(act_res_pattern)
        act_time_pattern=self.operate_time(act_time_pattern)
        
        act_attn,weights=self.act_attention(lstmout[0],act_pattern,act_pattern)
        res_attn,weights=self.res_attention(lstmout[1],act_res_pattern,act_res_pattern)
        time_attn,weights=self.time_attention(lstmout[2],act_time_pattern,act_time_pattern)

        # lsact=torch.cat((act_pattern.unsqueeze(dim=1),task_aggreation[:,0,:].unsqueeze(dim=1)),dim=1)
        # lsact_res=torch.cat((act_res_pattern.unsqueeze(dim=1),task_aggreation[:,1,:].unsqueeze(dim=1)),dim=1)
        # lsact_time=torch.cat((act_time_pattern.unsqueeze(dim=1),task_aggreation[:,2,:].unsqueeze(dim=1)),dim=1)
        # act_attn,weights=self.act_attention(lsact,lsact,lsact)
        # res_attn,weights=self.res_attention(lsact_res,lsact_res,lsact_res)
        # time_attn,weights=self.time_attention(lsact_time,lsact_time,lsact_time)

        # act_output=self.act_decoder(act_attn[:,-1,:])

        # act_out=self.act_decoder(act_attn[:,1,:])
        # res_out=self.res_decoder(res_attn[:,1,:])
        # time_out=self.time_decoder(time_attn[:,1,:])

        act_out=self.act_decoder(act_attn[:,-1,:])
        res_out=self.res_decoder(res_attn[:,-1,:])
        time_out=self.time_decoder(time_attn[:,-1,:])


        # act_out=self.act_decoder(act_attn[:,-1,:])
        # res_out=self.res_decoder(res_attn[:,-1,:])
        # time_out=self.time_decoder(time_attn[:,-1,:])



        return act_out,res_out,time_out


class Multi_view_multi_task_no_no(nn.Module):
    def __init__(self,embsize_list,num_list,hidden_dim,kernel_size,dropout,num_layers,pattern_embsize_list,pattern_num_list):
        super().__init__()
        self.embedding = MyEmbeddingLayer(embsize_list,num_list)
        # self.encoder=nn.Linear(sum(embsize_list[0:2])+2,32)

        self.tcn_layer=nn.ModuleList([TemporalConvNet(sum(embsize_list[0:2])+2,[hidden_dim]*num_layers, kernel_size=kernel_size, dropout=dropout) for i in range(3)])
        self.lstm_layer=nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim,batch_first=True) for i in range(3)])
        self.bn_layer=nn.ModuleList([nn.BatchNorm1d(hidden_dim) for i in range(3)])

        self.task_aggregation_layer=nn.MultiheadAttention(hidden_dim,4,batch_first=True)

        self.time_decoder=nn.Sequential(
            nn.Linear(hidden_dim,1)
        )

        self.act_pattern_embedding=nn.Embedding(num_list[0]+1,pattern_embsize_list[0],padding_idx=num_list[0])
        self.act_pattern_tcn=TemporalConvNet(embsize_list[0],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_res_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_res_res_pattern_embedding=nn.Embedding(pattern_num_list[1]+2,pattern_embsize_list[1],padding_idx=pattern_num_list[1]+1)
        self.act_res_pattern_tcn=TemporalConvNet(sum(pattern_embsize_list[0:2]),[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_res_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_res_pattern_bn=nn.BatchNorm1d(hidden_dim)

        self.act_time_act_pattern_embedding=nn.Embedding(pattern_num_list[0]+1,pattern_embsize_list[0],padding_idx=pattern_num_list[0])
        self.act_time_time_pattern_embedding=nn.Embedding(pattern_num_list[2]+2,pattern_embsize_list[2],padding_idx=pattern_num_list[2]+1)
        self.act_time_pattern_tcn=TemporalConvNet(pattern_embsize_list[0]+pattern_embsize_list[2],[hidden_dim]*num_layers,kernel_size=kernel_size,dropout=dropout)
        self.act_time_pattern_lstm=nn.LSTM(hidden_dim,hidden_dim,batch_first=True)
        self.act_time_pattern_bn=nn.BatchNorm1d(hidden_dim)



        self.act_attention=nn.MultiheadAttention(32,4,batch_first=True)
        self.act_decoder=nn.Linear(hidden_dim,num_list[0])

        self.res_attention=nn.MultiheadAttention(32,4,batch_first=True)
        self.res_decoder=nn.Linear(hidden_dim,num_list[1])

        self.time_attention=nn.MultiheadAttention(32,4,batch_first=True)
        self.time_decoder=nn.Linear(hidden_dim,1)

        self.tanh=nn.Tanh()

    def operate_res(self,act_res_pattern):
        act_embedding=self.act_res_act_pattern_embedding(act_res_pattern[:,:,0].long())
        res_embedding=self.act_res_res_pattern_embedding(act_res_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,res_embedding],dim=-1)
        tcnout=self.act_res_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_res_pattern_lstm(tcnout)[0]
        bn=self.act_res_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn
    
    def operate_time(self,act_time_pattern):
        act_embedding=self.act_time_act_pattern_embedding(act_time_pattern[:,:,0].long())
        time_embedding=self.act_time_time_pattern_embedding(act_time_pattern[:,:,1].long())
        embedding=torch.cat([act_embedding,time_embedding],dim=-1)
        tcnout=self.act_time_pattern_tcn(embedding.transpose(1,2)).transpose(1,2)
        lstmout=self.act_time_pattern_lstm(tcnout)[0]
        bn=self.act_time_pattern_bn(lstmout.contiguous().view(-1,lstmout.shape[2])).view(lstmout.shape)
        return bn

    def operate_act(self,act_pattern):
        pattern_embedding=self.act_pattern_embedding(act_pattern.long())
        pattern_tcn=self.act_pattern_tcn(pattern_embedding.transpose(1,2)).transpose(1,2)
        pattern_lstm=self.act_pattern_lstm(pattern_tcn)[0]
        bn_output=self.act_pattern_bn(pattern_lstm.contiguous().view(-1,pattern_lstm.shape[2])).view(pattern_lstm.shape)
        return bn_output

    def forward(self,data,act_pattern,act_res_pattern,act_time_pattern):
        embedding=self.embedding(data)
        # encoder=self.encoder(embedding)
        tcnout=[]
        for i in range(3):
            tcnout.append(self.tcn_layer[i](embedding.transpose(1, 2)).transpose(1, 2))
        lstmout=[]
        for i in range(3):
            # lstmout.append(self.lstm_layer[i](tcnout[i])[0])
            lstmout1=self.lstm_layer[i](tcnout[i])[0]
            # lstmout1=lstmout1.permute(0, 2, 1).contiguous()
            bn1=self.bn_layer[i](lstmout1.contiguous().view(-1,lstmout1.shape[2])).view(lstmout1.shape)
            # bn1=bn1.permute(0, 2, 1).contiguous()
            # lstmout.append(bn1[:,-1,:].unsqueeze(dim=1))
            lstmout.append(bn1)
            # lstmout.append(lstmout1[:,-1,:].unsqueeze(dim=1))

        # avgout=[]
        # for i in range(3):
        #     avgout.append(torch.mean(lstmout[i],axis=1).unsqueeze(dim=1))
        # lstmagg=torch.cat(avgout,dim=1)
        # # lstmagg=torch.cat(lstmout,dim=1)
        # task_aggreation,weights=self.task_aggregation_layer(lstmagg,lstmagg,lstmagg)
        
        # A,B,C=lstmout[0],lstmout[1],lstmout[2]
        # W=weights
        # A_prime = (A * W[:, 0, 0].view(-1, 1, 1) + B * W[:, 0, 1].view(-1, 1, 1) + C * W[:, 0, 2].view(-1, 1, 1))
        # B_prime = (A * W[:, 1, 0].view(-1, 1, 1) + B * W[:, 1, 1].view(-1, 1, 1) + C * W[:, 1, 2].view(-1, 1, 1))
        # C_prime = (A * W[:, 2, 0].view(-1, 1, 1) + B * W[:, 2, 1].view(-1, 1, 1) + C * W[:, 2, 2].view(-1, 1, 1))


        # act_pattern=self.operate_act(act_pattern)
        # act_res_pattern=self.operate_res(act_res_pattern)
        # act_time_pattern=self.operate_time(act_time_pattern)
        
        # act_attn,weights=self.act_attention(lstmout[0],act_pattern,act_pattern)
        # res_attn,weights=self.res_attention(lstmout[1],act_res_pattern,act_res_pattern)
        # time_attn,weights=self.time_attention(lstmout[2],act_time_pattern,act_time_pattern)

        # lsact=torch.cat((act_pattern.unsqueeze(dim=1),task_aggreation[:,0,:].unsqueeze(dim=1)),dim=1)
        # lsact_res=torch.cat((act_res_pattern.unsqueeze(dim=1),task_aggreation[:,1,:].unsqueeze(dim=1)),dim=1)
        # lsact_time=torch.cat((act_time_pattern.unsqueeze(dim=1),task_aggreation[:,2,:].unsqueeze(dim=1)),dim=1)
        # act_attn,weights=self.act_attention(lsact,lsact,lsact)
        # res_attn,weights=self.res_attention(lsact_res,lsact_res,lsact_res)
        # time_attn,weights=self.time_attention(lsact_time,lsact_time,lsact_time)

        # act_output=self.act_decoder(act_attn[:,-1,:])

        # act_out=self.act_decoder(act_attn[:,1,:])
        # res_out=self.res_decoder(res_attn[:,1,:])
        # time_out=self.time_decoder(time_attn[:,1,:])

        # act_out=self.act_decoder(act_attn[:,-1,:])
        # res_out=self.res_decoder(res_attn[:,-1,:])
        # time_out=self.time_decoder(time_attn[:,-1,:])


        act_out=self.act_decoder(lstmout[0][:,-1,:])
        res_out=self.res_decoder(lstmout[1][:,-1,:])
        time_out=self.time_decoder(lstmout[2][:,-1,:])



        return act_out,res_out,time_out
