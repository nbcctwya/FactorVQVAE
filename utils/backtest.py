import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from pathlib import Path

from module.transformer import AutoRegressiveTransformer
import math
from data.dataset import init_data_loader
import pandas as pd 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from utils import seed_everything
from qlib.data.dataset import DatasetH, TSDatasetH, DataHandlerLP, TSDataSampler

class InvestmentModel:
    def __init__(self, config, model_path=None, seed=42, num_workers=0):
        seed_everything(seed)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model(model_path) if model_path is not None else self.build_model()
        self.model.feature_extractor.stage = 2
        self.model.to(self.device)
        self.model.eval()
        self.num_workers = num_workers
        
    def build_model(self, saved_model_path=None):
        saved_model_path = self.config['transformer']['saved_model'] if saved_model_path is None else saved_model_path
        checkpoint = torch.load(
            saved_model_path, map_location=self.device, weights_only=False
        )['state_dict']
        
        model = AutoRegressiveTransformer(temperature= 1, config = self.config)
        state_dict = {k.replace('mingpt.', ''): v for k, v in checkpoint.items() if k.startswith('mingpt.')}
       
        model.load_state_dict(state_dict)
        
        return model
    
    def build_data_loader(self, data_path):
        print("Load data from {}".format(data_path))

        self.dataframe = pd.read_pickle(data_path)
        handlerlp = DataHandlerLP.from_df(self.dataframe)
        dic = {
            'train' : self.config['data']['train_period'],
            'valid' : self.config['data']['valid_period'],
            'test' : self.config['data']['test_period']
        }
        # todo: check here 
        TsDataset = TSDatasetH(handler = handlerlp, segments = dic, step_len = self.config['data']['window_size'])
        test_prepare = TsDataset.prepare(segments = 'test', data_key=DataHandlerLP.DK_I) # DK_I: inference data # but we have to use DK_L
        test_index = test_prepare.get_index()
        self.dataloader = self.get_dataloader(test_prepare, self.num_workers)

        return test_index
    
    def get_dataloader(self, handler, num_workers):
        dataloader = init_data_loader(handler, shuffle=False, num_workers=num_workers)
        return dataloader
    
    def inference(self, data_path, top_k=3):
        return self.inference_gpt(data_path, top_k)

    @torch.no_grad()
    def inference_gpt(self, data_path, top_k=3):
        
        test_index = self.build_data_loader(data_path)
        pred = []
        real = []
        loss = []
        z_e_list = []
        print("Start inference")
        for batch in tqdm(self.dataloader):
            batch = batch.squeeze(0)
            num_features = self.config['vqvae']['num_features']
            num_market = self.config['vqvae']['market_features']
            firm_char = batch[:, :, :num_features].to(self.device)
            market = batch[:, :, num_features:num_features + num_market].to(self.device)
            model_label_col = num_features + num_market
            labels = batch[:, :, model_label_col].unsqueeze(-1).to(self.device)
            delay = self.model.label_delay
            known_labels = labels[:, :-delay, :]
            if batch.shape[-1] > model_label_col + 1:
                y = batch[:, -1, -1].unsqueeze(-1).to(self.device)
            else:
                y = labels[:, -1, :]

            # Use the same leakage-safe path as training and test inference.
            _, y_hat = self.model.predict(firm_char, known_labels, market)
            y_hat = y_hat[:, -1, :]
            reconstr_loss = F.mse_loss(y_hat, y)

            pred.append(y_hat.cpu().detach().numpy())
            real.append(y.cpu().detach().numpy())
            loss.append(reconstr_loss.cpu().detach().numpy())
            z_e_list.append(self.model.encoder(known_labels).cpu().detach().numpy())

        pred = pd.Series(np.concatenate(pred, axis=0).squeeze(), index=test_index)
        real = pd.Series(np.concatenate(real, axis=0).squeeze(), index=test_index)
        loss = np.mean(loss)

        return pred, real, loss, z_e_list

    @torch.no_grad()
    def check_tokenizer(self, data_path):
        raise NotImplementedError(
            "Legacy tokenizer audit is incompatible with delayed labels; "
            "use AutoRegressiveTransformer.predict with labels through t-2."
        )

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out
    

    @torch.no_grad()
    def inference_bert(self, data_path):
        raise NotImplementedError(
            "The legacy BERT path does not implement delayed-label inference."
        )
    

def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.mean(ric_values_multiindex)
    std = np.std(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def Cal_IC_IR(df, column1='LABEL0', column2='Pred'):
    ic = []
    ric = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0'] = daily_data[column1]
        daily_data['pred'] = daily_data[column2]
        ic_, ric_ = calc_ic(daily_data['pred'], daily_data['LABEL0'])
        ic.append(ic_)
        ric.append(ric_)

    metrics = {
        'IC': np.mean(ic),
        'IC_IR': np.mean(ic) / np.std(ic),
        'RankIC': np.mean(ric),
        'RankIC_IR': np.mean(ric) / np.std(ric)
    }

    return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    


def calculate_table_metrics(series, period, name, target_return=0):

    if period is not None:
        if type(period) == int:
            series = series[series.index.year == int(period)].copy()
            # series['return'] = series['return'] / series['return'].iloc[0]  
        elif type(period) == list:
            series = series.loc[period[0]:period[1]].copy()
    try:  
        daily_log_returns = series['return']
        cum_return = series['return'].cumsum()
    except:
        daily_log_returns = series
        cum_return = series.cumsum()
    normal_cum_return = np.exp(cum_return)
    
    # MDD 계산을 위해 누적 일반 리턴 사용
    max_cumulative_returns = normal_cum_return.cummax()
    drawdown = (normal_cum_return - max_cumulative_returns) / (max_cumulative_returns + 1e-9) 
    mdd = drawdown.min()

    # 연간 수익률 및 기타 지표 계산
    annual_return = daily_log_returns.mean() * 252
    annual_std = daily_log_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std

    # Sortino Ratio
    # Calculate downside deviation
    downside_returns = daily_log_returns[daily_log_returns < target_return]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - target_return) / downside_std if downside_std != 0 else np.nan
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(mdd) if mdd != 0 else np.nan

    # Turnover
    turnover = series['turnover'].mean()
    turnover = round(turnover, 4)
    
    result = {
        'Annualized Return': round(annual_return, 4),
        'Annual Std': round(annual_std, 4),
        'Sharpe Ratio': round(sharpe_ratio, 4),
        'Sortino Ratio': round(sortino_ratio, 4),
        'Calmar Ratio': round(calmar_ratio, 4),
        'MDD': round(mdd, 4),
        'Cumulative Returns': round(cum_return.iloc[-1], 4),
        'Turnover': turnover
    }

    return pd.DataFrame.from_dict(result, orient='index', columns=[f'{name}'])
