import numpy as np
import pandas as pd
from tabulate import tabulate


def pprint_df(df, tablefmt='psql'):
    print(tabulate(df, headers='keys', tablefmt=tablefmt))


if __name__ == "__main__":
    all_datasets = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                    'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']
    records = []
    for dataset_name in sorted(all_datasets, key=lambda v: v.upper()):
        edge_raw_features = np.load('../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
        edge_csv = pd.read_csv('../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
        print("\nDataset: {}".format(dataset_name))
        unique_ts = np.sort(edge_csv['ts'].unique())
        unique_count = unique_ts.shape[0]
        min_delta = np.diff(unique_ts).min()
        t_min = edge_csv['ts'].min()
        t_max = edge_csv['ts'].max()
        time_span = t_max - t_min
        
        print("  Time span: {} to {} (delta {})".format(t_min, t_max, min_delta))
        print("  Unique timestamps: {}".format(unique_count))
       
        info = {'dataset_name': dataset_name,
                'num_nodes': node_raw_features.shape[0] - 1,
                'node_feat_dim': node_raw_features.shape[-1],
                'num_edges': edge_raw_features.shape[0] - 1,
                'edge_feat_dim': edge_raw_features.shape[-1]}
        records.append(info)

    info_df = pd.DataFrame.from_records(records)
    pprint_df(info_df)
