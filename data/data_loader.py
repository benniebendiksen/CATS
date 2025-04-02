import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from utils.tools import parse_ratios
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

    
    
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',ratios=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            #print(df_data.values.shape)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',ratios=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Crypto(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='btcusdc_pca_components_44_proper_split.csv',
                 target='close', scale=True, timeenc=0, freq='15min', ratios=None):
        """
        Dataset for cryptocurrency price forecasting with PCA components

        Args:
            root_path: root path of the data file
            flag: 'train', 'test' or 'val'
            size: [seq_len, label_len, pred_len]
            features: 'M', 'MS' or 'S'
            data_path: data file
            target: target column (e.g., 'close')
            scale: whether to scale the data
            timeenc: time encoding method
            freq: time frequency
            ratios: train/val/test split ratios
        """
        if size == None:
            self.seq_len = 96  # 24 hours (96 * 15 min)
            self.label_len = 48
            self.pred_len = 1  # Predict 1 step ahead for binary classification
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']

        # Store parameters
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.ratios = ratios or [0.7, 0.1, 0.2]  # Default ratios if not provided

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        # Only print detailed information for the training dataset to avoid redundancy
        self.is_train = (flag == 'train')

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Store dataset properties
        self.total_rows = len(df_raw)
        self.has_split_column = 'split' in df_raw.columns

        # Process data splits
        if self.has_split_column:
            # Use the split column to filter data
            train_data = df_raw[df_raw['split'] == 'train']
            val_data = df_raw[df_raw['split'] == 'val']
            test_data = df_raw[df_raw['split'] == 'test']

            # Set the appropriate dataset based on flag
            if self.flag == 'train':
                active_data = train_data
            elif self.flag == 'val':
                active_data = val_data
            else:  # 'test'
                active_data = test_data

            # for debugging
            if self.flag == 'test':
                print(f"Test data: {len(test_data)} samples, {len(test_data) / len(df_raw):.2%} of total")
                pos_count = (test_data[self.target].shift(-self.pred_len) > test_data[self.target]).sum()
                print(f"Test data positive examples: {pos_count}, {pos_count / len(test_data):.2%}")

            # Define borders
            border1 = 0
            border2 = len(active_data)

            # Store split info
            self.split_info = {
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            }

            # Save indices for integrity checking
            self.active_indices = active_data.index.tolist()

            # Drop split column before feature processing
            df_raw = df_raw.drop(columns=['split'])
        else:
            # Traditional percentage-based split
            num_train = int(len(df_raw) * self.ratios[0])
            num_test = int(len(df_raw) * self.ratios[2])
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

            # Map set_type to numeric index
            type_map = {'train': 0, 'val': 1, 'test': 2}
            set_type_idx = type_map[self.flag]

            border1 = border1s[set_type_idx]
            border2 = border2s[set_type_idx]

            active_data = df_raw

            # Store split info
            self.split_info = {
                'train': num_train,
                'val': num_vali,
                'test': num_test
            }

        # Create binary labels for price change prediction
        close_prices = df_raw[self.target].values
        binary_labels = np.zeros((len(close_prices), 1))

        # For each potential sequence starting point
        for i in range(len(close_prices) - self.seq_len - self.pred_len):
            # Calculate prediction point index
            pred_idx = i + self.seq_len

            # Get prices at prediction point and future point
            pred_price = close_prices[pred_idx]
            future_price = close_prices[pred_idx + self.pred_len]

            # Set label based on future price direction from prediction point
            binary_labels[i, 0] = 1.0 if future_price > pred_price else 0.0

        # Create a mapping from sequence indices to original indices
        self.sequence_indices = {}
        for i in range(len(self.active_indices) - self.seq_len - self.pred_len + 1):
            # Skip if we don't have enough data for this sequence
            if i >= len(self.active_indices) or i + self.seq_len >= len(self.active_indices):
                continue

            orig_start_idx = self.active_indices[i]
            pred_idx = orig_start_idx + self.seq_len
            future_idx = pred_idx + self.pred_len

            # Skip if we'd go beyond the end of the original data
            if future_idx >= len(close_prices):
                continue

            # Calculate price at prediction point and future point
            pred_price = close_prices[pred_idx]
            future_price = close_prices[future_idx]

            # Calculate label directly
            calculated_label = 1.0 if future_price > pred_price else 0.0

            # Store all relevant information
            self.sequence_indices[i] = {
                'orig_start_idx': orig_start_idx,
                'pred_idx': pred_idx,
                'future_idx': future_idx,
                'pred_price': pred_price,
                'future_price': future_price,
                'price_change': (future_price - pred_price) / pred_price * 100.0,
                'label': calculated_label
            }

        # Feature columns
        if self.features == 'M' or self.features == 'MS':
            cols = list(df_raw.columns)
            # Remove target and date
            if self.target in cols:
                cols.remove(self.target)
            if 'date' in cols:
                cols.remove('date')

            # Save feature list
            self.feature_list = cols

            # Extract appropriate data
            df_data = df_raw[cols]
            active_df_data = active_data[cols]
            train_df_data = train_data[cols]

            # Always extract entire dataset for scaling properly
            # df_data_full = df_raw[cols + [self.target]]
            # # Extract training data for fitting scaler
            # train_df_data = df_raw[df_raw['split'] == 'train'][cols + [self.target]] if self.has_split_column else \
            #     df_raw.iloc[:int(len(df_raw) * self.ratios[0])][cols + [self.target]]
            #
            # # Extract appropriate data
            # if self.has_split_column:
            #     df_data = active_data[cols + [self.target]]
            #     # train_df_data = train_data[cols + [self.target]]
            # else:
            #     df_data = df_raw[cols + [self.target]]
            #     # train_df_data = df_raw.iloc[:self.split_info['train']][cols + [self.target]]
        elif self.features == 'S':
            if self.has_split_column:
                df_data = active_data[[self.target]]
                train_df_data = train_data[[self.target]]
            else:
                df_data = df_raw[[self.target]]
                train_df_data = df_raw.iloc[:self.split_info['train']][[self.target]]
        else:
            raise ValueError(f"Invalid features argument: {self.features}")

        # Scale the features using ONLY the training data
        if self.scale:
            print(f"Fitting scaler on {len(train_df_data)} training samples")
            # Always fit scaler on training data only
            self.scaler.fit(train_df_data.values)
            # Store scaling stats
            if self.flag == 'train':
                print(f"Scaler mean range: [{np.min(self.scaler.mean_):.4f}, {np.max(self.scaler.mean_):.4f}]")
                print(
                    f"Scaler std range: [{np.min(np.sqrt(self.scaler.var_)):.4f}, {np.max(np.sqrt(self.scaler.var_)):.4f}]")

            # Transform only the active data subset
            data = self.scaler.transform(df_data.values)

        # # Scale the features
        # if self.scale:
        #     # Always fit scaler on training data only
        #     self.scaler.fit(train_df_data.values)
        #     # Transform all data
        #     data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process timestamps
        if 'date' in df_raw.columns:
            if self.has_split_column:
                # For the new split method, get timestamps from active dataset
                df_stamp = active_data[['date']].reset_index(drop=True)
            else:
                # For traditional split method, get timestamps from the window
                df_stamp = df_raw[['date']][border1:border2]

            # Convert to datetime
            df_stamp['date'] = pd.to_datetime(df_stamp.date)

            # Create time features
            if self.timeenc == 0:
                # Manual time encoding
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                # Automatic time features
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_stamp), 1))
        else:
            # If no date column, create empty timestamp features
            data_stamp = np.zeros((len(df_data), 1))

        # Store processed data
        if self.has_split_column:
            # Extract from the active dataset
            # self.data_x = data
            # For split based method, extract from the active dataset
            self.data_x = self.scaler.transform(active_df_data.values)
            # Get corresponding binary labels for active dataset
            active_indices = active_data.index.tolist()
            self.data_y = binary_labels[active_indices]
            # self.data_y = binary_labels[self.active_indices] if hasattr(self, 'active_indices') else \
            #     binary_labels[border1:border2]
        else:
            # Use the window
            self.data_x = data[border1:border2]
            self.data_y = binary_labels[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """Get a single sample"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # Get input sequence
        seq_x = self.data_x[s_begin:s_end]

        # Get target sequence
        seq_y = np.zeros((self.label_len + self.pred_len, 1))

        # Fill history part of target
        if r_begin >= 0 and r_begin < len(self.data_y):
            available_len = min(self.label_len, len(self.data_y) - r_begin)
            seq_y[:available_len, 0] = self.data_y[r_begin:r_begin + available_len, 0]

        # Fill prediction part of target
        future_begin = r_begin + self.label_len
        if future_begin < len(self.data_y):
            # For test set, use sequence mapping to ensure correct label if available
            if self.flag == 'test' and hasattr(self, 'sequence_indices') and index in self.sequence_indices:
                # Use the stored label for the prediction part
                seq_y[self.label_len:, 0] = self.sequence_indices[index]['label']
            else:
                # Standard behavior with safety checks
                available_len = min(self.pred_len, len(self.data_y) - future_begin)
                seq_y[self.label_len:self.label_len + available_len, 0] = self.data_y[
                                                                          future_begin:future_begin + available_len, 0]

        # Get timestamp features
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end] if r_begin >= 0 and r_end <= len(self.data_stamp) else np.zeros(
            (self.label_len + self.pred_len, self.data_stamp.shape[1]))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """Return the effective length of the dataset after accounting for sequence windows"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse transform scaled data back to original scale"""
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',ratios=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.ratios=parse_ratios(ratios)

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * self.ratios[0])
        num_test = int(len(df_raw) * self.ratios[2])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min',ratios=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.ratios=parse_ratios(ratios)
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
