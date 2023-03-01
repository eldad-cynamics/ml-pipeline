import numpy as np
import pandas as pd
from datetime import timedelta, date
from autoencoder_model.src.dataset.create_dataset_utils import query_redshift, create_X_v1_v3
from loguru import logger

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield (start_date + timedelta(n)).strftime("%Y-%m-%d")

if __name__ == '__main__':

    client_devices_path = '../data/clients_devices.csv'
    clients_devices = pd.read_csv(f'{client_devices_path}')
    clients_devices = clients_devices[clients_devices.status == 'active']

    days = list(daterange(date(2022, 2, 13), date(2022, 3, 15)))

    clients_devices = [
        ('Walla', 'VTPocketASA', 1 / 100, '60994789ee5d600009a40290', '6189947d22b29b0009b3b106'),
        ('Guilford',1,0.01,'619807593896d10009e0b2ad','61981d0b9f156d000a8bf356'),
        ('Walla', 'VTCLDUOASA', 1/100, '60994789ee5d600009a40290', '618996f5f7b6af0009990482'),
        ('Walla', 1, 1/1000,      '60994789ee5d600009a40290', '609abe351175bd00090a92fd')
    ]


    for client, device, sr, account_id, device_id in clients_devices:
        logger.info(f"{client}, {device}, {sr}, {account_id}, {device_id}")
        for day in days:
            logger.info(f"{day}")
            try:
                df = query_redshift(client, device, sr, day, account_id, device_id)
                tdf = df.groupby('time_window').agg({'flowPackets': 'sum'})
                print(np.mean(list(tdf['flowPackets'].values) + [0] * (17280 - len(tdf['flowPackets'].values))))
                print("less than a day: "+str(np.mean(list(tdf['flowPackets'].values))))
            except Exception as e:
                logger.exception(e)

    # In[ ]:


    (df.time_window.max() - df.time_window.min())/5


    # In[7]:


    tdf.describe()


    # In[6]:


    import gzip
    import boto3
    import numpy as np
    import pandas as pd

    from io import BytesIO, TextIOWrapper
    from sqlalchemy import create_engine
    import psycopg2


    # In[7]:


    import gzip
    import boto3
    import numpy as np
    import pandas as pd

    from io import BytesIO, TextIOWrapper
    from sqlalchemy import create_engine
    import psycopg2

    dbname='cynamics'
    cluster='prodredshiftcluster.chk0levkctdj.us-east-1'
    port=5439
    # user='superadmin'
    # password='usY45Np8SaV1^f9'
    user='ro_user'
    password='37gvWDy5GZnwwdXjhGgQ'



    connstr = f'postgresql://{user}:{password}@{cluster}.redshift.amazonaws.com:5439/cynamics'
    engine = create_engine(connstr)
    connection = psycopg2.connect(dbname=dbname, host=f'{cluster}.redshift.amazonaws.com',
                                  port='5439', user=user, password=password)

    client, account_id, device_id = 'Eugene', '5ed4e7c5403dc20018b47e6c', '5f1b1cbd5a19bc0011aa66ff'
    client, account_id, device_id = 'DouglasCounty', '5f15d4502997970011989705', '5f1eee725a19bc0011aa73dd'
    client, account_id, device_id = 'WMU', '5ec7da694d4a86001106da83', '5ec7db437f736700114636a9'
    client, account_id, device_id = 'Restorepoint', '5f53901a4e0ffa001a08ba2e', ''


    # In[ ]:


    start_time = 1597752000
    column, value = 'sourceport', 443

    query = f"""
    WITH temp AS (
        SELECT extract(epoch from creationtime) - MOD(extract(epoch from creationtime), 5) as time_window, sourceport, SUM(flowpackets) as sum
        FROM public.rawdatas
        WHERE accountid='{account_id}'
          AND deviceid='{device_id}'
          AND extract(epoch from creationtime)>={start_time}
          AND extract(epoch from creationtime)<={start_time+3600}
        GROUP BY 1,2
    )
    SELECT sourceport, MAX(sum)
    FROM temp
    GROUP BY 1
    """

    query = f"""
    SELECT ipv4src, ipv4dest, SUM(flowpackets) as flowpackets
    FROM public.rawdatas
    WHERE accountid='{account_id}'
      AND deviceid='{device_id}'
      AND extract(epoch from creationtime)>={start_time}
      AND extract(epoch from creationtime)<={start_time}+3600
      AND {column}={value}
    GROUP BY 1,2
    """

    query = f"""
    SELECT left(creationtime, 10), deviceid, samplingrate, count(*)
    FROM public.rawdatas
    WHERE accountid='{account_id}'
      ---AND deviceid='{device_id}'
    GROUP BY 1,2,3
    """

    # with connection.cursor() as cursor:
    #     cursor.execute(query)
    #     result = cursor.fetchall()

    with engine.connect() as conn, conn.begin():
        query_df = pd.read_sql(query, conn)
    query_df.sort_values(['deviceid', 'left'])


    # In[ ]:


    device_id


    # In[ ]:


    with connection.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
    result


    # In[ ]:


    df[df.left == '2020-09-11']

    df[['ipv4src', 'ipv4dest']] = np.sort(df[['ipv4src', 'ipv4dest']])
    df.groupby(['ipv4src', 'ipv4dest']).sum().reset_index().sort_values('flowpackets', ascending=False)
    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:





    # In[ ]:




