import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np 
import logging
import pandas as pd
import copy
from torch.utils.data.dataloader import default_collate

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

        self.index_df = self.data_source.get_index()
        datetime_level = self.index_df.names.index('datetime') # 'datetime' 레벨 위치 찾기
        daily_groups = pd.Series(self.index_df.values).groupby(self.index_df.get_level_values(datetime_level))

        self.daily_count = daily_groups.size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        # 날짜 순서 보장을 위해 unique dates 저장
        self.dates = daily_groups.groups.keys() # 정렬된 날짜 리스트

    def __iter__(self):
        date_indices = np.arange(len(self.dates))
        if self.shuffle:
            np.random.shuffle(date_indices)

        # 전체 데이터셋에서의 실제 인덱스 번호(정수 위치)를 yield 해야 함
        # self.index_df 를 기준으로 각 날짜에 해당하는 정수 인덱스를 찾아야 함
        datetime_level = self.index_df.names.index('datetime')
        all_datetimes = self.index_df.get_level_values(datetime_level)

        for i in date_indices:
            target_date = list(self.dates)[i] # 접근 방식 수정 필요할 수 있음 (dates 타입 확인)
            # 해당 날짜를 가진 모든 샘플의 *정수 위치* 인덱스 찾기
            indices_for_date = np.where(all_datetimes == target_date)[0]
            if len(indices_for_date) != self.daily_count[i]:
                print(f"Warning: Index count mismatch for date {target_date}. Expected {self.daily_count[i]}, Found {len(indices_for_date)}")
            yield indices_for_date # 해당 날짜의 인덱스 배열 자체를 yield

    def __len__(self):
        return len(self.daily_count) # len(self.data_source)
 

def init_data_loader(handler, shuffle, num_workers=0):
    sampler = DailyBatchSamplerRandom(handler, shuffle)
    num_batches_per_epoch = len(sampler)
    # 모든 데이터를 float 타입으로 변환하는 collate 함수
    def float_collate_fn(batch):
        batch = default_collate(batch)
        if isinstance(batch, torch.Tensor):
            return batch.float()
        return batch
    
    data_loader = DataLoader(handler,
                             sampler=sampler,
                             pin_memory=True,
                             num_workers=num_workers,
                             drop_last=False,
                             collate_fn=float_collate_fn)  # float 변환 collate 함수 적용
    
    return data_loader