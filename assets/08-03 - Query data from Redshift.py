import numpy as np
import pandas as pd
from datetime import timedelta, date
from src.dataset.create_dataset_utils import query_redshift, create_X_v1_v3,redshift_query,save_csv_gz_to_s3
from loguru import logger
from pathlib import Path
from datetime import datetime


def daterange (start_date: date, end_date: date) -> str:
    for n in range(int((end_date - start_date).days)):
        yield (start_date + timedelta(n)).strftime("%Y-%m-%d")


def get_clients_metadata (client_devices_path: Path) -> pd.DataFrame:
    clients_devices = pd.read_csv(f'{client_devices_path}')
    clients_devices = clients_devices[clients_devices.status == 'active']
    return clients_devices


if __name__ == '__main__':

    client_devices = get_clients_metadata(client_devices_path='../data/clients_devices.csv')
    clients_devices = [
        ('Sioux', 1, 1 / 1000, '60a52c5d7c0df300090597c2', '60a66eae1265d10009c77766'),
    ]

    start_date,end_date = date(2023, 2, 3), date(2023, 2, 5)
    days = list(daterange(start_date, end_date))
    time_window_size = 5

    for client, device, sr, account_id, device_id in clients_devices:
        logger.info(f"{client}, {device}, {sr}, {account_id}, {device_id}")
        for day in days:
            logger.info(f"{day}")
            print(f'Querying {client}, device={device}, day={day}', end='...')
            start = (datetime.strptime(day, '%Y-%m-%d') - datetime(1970, 1, 1)).total_seconds()
            end = start + (24 * 60 * 60)
            query_order = f"""
            select creationtime, ipv4src, sourceport, ipprotocol, ipv4dest, destport, length, flowpackets
            from public.rawdatas
            where accountid='{account_id}'
              and deviceid = '{device_id}'
              and creationtime >= timestamp 'epoch' + {start} * interval '1 second'
              and creationtime <  timestamp 'epoch' + {end}   * interval '1 second'
              and samplingrate = {sr}
            --limit 100"""
            try:
                df = redshift_query(query_order)
                print('df shape: ', df.shape, end='...')

                df['timestamp'] = (pd.to_datetime(df['creationtime']) - pd.Timestamp("1970-01-01")) // pd.Timedelta(
                    '1s')
                df['time_window'] = df['timestamp'] - (df['timestamp'] % time_window_size)

                print('Renaming', end='...')
                df = df.rename(columns={'length': 'volume',
                                        'ipv4src': 'ipv4Src',
                                        'sourceport': 'sourcePort',
                                        'ipprotocol': 'ipProtocol',
                                        'ipv4dest': 'ipv4Dest',
                                        'destport': 'destPort',
                                        'flowpackets': 'flowPackets'
                                        })
                df = df.groupby(['ipv4Src', 'sourcePort', 'ipProtocol',
                                 'ipv4Dest', 'destPort', 'time_window']) \
                    .agg({'flowPackets': sum,
                          'volume': sum
                          }).sort_values('time_window').reset_index()

                save_csv_gz_to_s3(df, f'clients/{client}/sr={sr}/device={device}/sampled/{day}.csv.gz')
                tdf = df.groupby('time_window').agg({'flowPackets': 'sum'})
                logger.info(np.mean(list(tdf['flowPackets'].values) + [0] * (17280 - len(tdf['flowPackets'].values))))
                logger.info("less than a day: " + str(np.mean(list(tdf['flowPackets'].values))))
            except Exception as e:
                logger.exception(e)
