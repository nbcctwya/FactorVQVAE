import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import spearmanr

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
        'ICIR': np.mean(ic) / np.std(ic),
        'RankIC': np.mean(ric),
        'RankICIR': np.mean(ric) / np.std(ric)
    }

    return metrics
    # return pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])



@torch.no_grad()
def run_inference(model, data_loader, device='cuda'):

    model.eval()
    model.to(device)
    preds = []
    reals = []

    test_index = data_loader.dataset.get_index().sortlevel(0)[0]

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running Inference")):
        batch = batch.to(device)
        batch = batch.squeeze(0)    
        num_features = model.config['vqvae']['num_features']
        market_features = model.config['vqvae']['market_features']
        firm_char = batch[:, :, :num_features]
        model_label_col = num_features + market_features
        inputs = batch[:, :, model_label_col].unsqueeze(-1)
        market = batch[:, :, num_features:num_features + market_features]

        delay = model.mingpt.label_delay
        known_inputs = inputs[:, :-delay, :]
        # A 173-column test cache carries raw return as the final column for
        # metrics.  The normalized model label above alone generates tokens.
        y = batch[:, -1, -1] if batch.shape[-1] > model_label_col + 1 else inputs[:, -1, 0]
        _, y_hat = model.mingpt.predict(firm_char, known_inputs, market)
        y_hat = y_hat[:, -1, :]

        preds.append(y_hat.cpu().detach().numpy())
        reals.append(y.cpu().detach().numpy())

    preds = pd.Series(np.concatenate(preds, axis=0).squeeze(), index=test_index)
    reals = pd.Series(np.concatenate(reals, axis=0).squeeze(), index=test_index)
    df = pd.DataFrame({'score': preds, 'label': reals})

    rankic = RankIC(df.dropna(), column1='score', column2='label')
    print(f"RankIC: {rankic}")
    icir = Cal_IC_IR(df.dropna(), column1='label', column2='score')
    print(f"Metrics: {icir}")

    return df, rankic, icir
