import numpy as np
import pandas as pd
from datetime import timedelta, date
from autoencoder_model.src.dataset.create_dataset_utils import query_redshift, create_X_v1_v3
from loguru import logger
from pathlib import Path


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

    start_date,end_date = date(2023, 2, 1), date(2023, 2, 3)
    days = list(daterange(start_date, end_date))


    for client, device, sr, account_id, device_id in clients_devices:
        logger.info(f"{client}, {device}, {sr}, {account_id}, {device_id}")
        for day in days:
            logger.info(f"{day}")
            try:
                df = query_redshift(client, device, sr, day, account_id, device_id)
                tdf = df.groupby('time_window').agg({'flowPackets': 'sum'})
                logger.info(np.mean(list(tdf['flowPackets'].values) + [0] * (17280 - len(tdf['flowPackets'].values))))
                logger.info("less than a day: " + str(np.mean(list(tdf['flowPackets'].values))))
            except Exception as e:
                logger.exception(e)
