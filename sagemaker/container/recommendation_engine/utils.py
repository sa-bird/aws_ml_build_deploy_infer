import boto3
import snowflake.connector
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import dsa
from cryptography.hazmat.primitives import serialization
import pandas as pd

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



def load_configs(config_type, CONFIGS_DIR='bin/configs'):
    if config_type=='snowflake':
        filename= 'snowflake.yml'

    snowflake_configs_fp = os.path.join(CONFIGS_DIR, filename)
    stream = open(snowflake_configs_fp, 'r+')
    snowflake_configs = yaml.load(stream, Loader=Loader)
    return snowflake_configs

def get_data_in_pd(cur, sql):
    cur.execute(sql)
    all_rows = cur.fetchall()
    num_fields = len(cur.description)
    field_names = [i[0] for i in cur.description]
    df = pd.DataFrame(all_rows)
    df.columns = field_names
    return df



class SnowFlake_Connector:
    def __init__(self, warehouse, database, schema, role, region='us-west-2') -> None:
        self.warehouse= warehouse
        self.database = database
        self.schema = schema
        self.role = role
        self.region = region

    def get_credentials(self, params):
        ssm = boto3.client('ssm',self.region)
        response = ssm.get_parameters(
            Names=params,
            WithDecryption=True
        )
            
        # Build dict of credentials
        param_values={k['Name']:k['Value'] for k in  response['Parameters']}
        return param_values
    
    def get_cursor(self):
        params=['/SNOWFLAKE/SAGEMAKER_DEV_USERNAME'
        ,'/SNOWFLAKE/ACCOUNT_ID'
        ,'/SNOWFLAKE/SAGEMAKER_DEV_USER_PHRASE'
        ,'/SAGEMAKER/NOTEBOOK/KEY_VALUE'
       ]
        param_values= self.get_credentials(params)

        # with open(param_values['/SAGEMAKER/NOTEBOOK/KEY_VALUE'], "rb") as key:
        #     p_key= serialization.load_pem_private_key(
        #         key.read(),
        #         password=param_values['/SNOWFLAKE/SAGEMAKER_DEV_USER_PHRASE'].encode(),
        #         backend=default_backend()
        #     )

        key = param_values['/SAGEMAKER/NOTEBOOK/KEY_VALUE']
        # print(f'key is {key}')

        p_key= serialization.load_pem_private_key(
                key.encode('ascii'),
                password=param_values['/SNOWFLAKE/SAGEMAKER_DEV_USER_PHRASE'].encode(),
                backend=default_backend()
            )

        # print(param_values['/SNOWFLAKE/SAGEMAKER_DEV_USER_PHRASE'])

        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption())

        ctx = snowflake.connector.connect(
            user=param_values['/SNOWFLAKE/SAGEMAKER_DEV_USERNAME'],
            account=param_values['/SNOWFLAKE/ACCOUNT_ID'],
            private_key=pkb,
            warehouse=self.warehouse,
            database=self.warehouse,
            schema=self.schema,
            role=self.role
            )

        return ctx.cursor()
    
    def execute_query(self, query):
        cur = self.get_cursor()
        print(f'cursor returned')
        
        try:
            cur.execute(query)
            data=cur.fetch_pandas_all()
            cur.close()
        finally:
            cur.close()
        
        return data

if __name__ == '__main__':
    import pandas as pd
    input_table_name= 'CUSTOMER360_DATA_SCIENCE_DEV.C360_SCRATCH.ratings_train_data'
    
    sf_configs = load_configs('snowflake')
    
    sc = SnowFlake_Connector(
        warehouse=sf_configs['WAREHOUSE'],
        database=sf_configs['DATABASE'],
        schema=sf_configs['SCHEMA'],
        role=sf_configs['ROLE']
    )
    
    # get the training data from Snowflake
    cur= sc.get_cursor()

    sql =  "select * from " + input_table_name
    df = get_data_in_pd(cur, sql)

    print(df.head())
    cur.close()

