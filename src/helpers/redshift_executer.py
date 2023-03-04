from src.helpers.config_loader import get_param
from psycopg2.extensions import QueryCanceledError
import psycopg2
from src.helpers import logger

REDSHIFT_PASSWORD = get_param("REDSHIFT_PASSWORD")
REDSHIFT_USERNAME = get_param("REDSHIFT_USERNAME")
REDSHIFT_PORT = '5439'
REDSHIFT_HOST = get_param("REDSHIFT_HOST")
REDSHIFT_DB_NAME = get_param("REDSHIFT_DB_NAME")

# Query timeout in milliseconds
QUERY_TIMEOUT = 1000 * 60 * 13


def execute_query(query, params):
    connection = None
    try:
        result = []
        logger.info('Trying to connect to redshift', None, redshift_uri=REDSHIFT_HOST)
        connection = psycopg2.connect(dbname=REDSHIFT_DB_NAME, host=REDSHIFT_HOST, port=REDSHIFT_PORT, user=REDSHIFT_USERNAME, password=REDSHIFT_PASSWORD, options=f'-c statement_timeout={QUERY_TIMEOUT}')
        with connection.cursor() as cursor:
            redshift_query = cursor.mogrify(query, params)
            logger.info("Querying redshift", None, redshift_query=str(redshift_query))
            cursor.execute(query, params)
            result = cursor.fetchall()
    except QueryCanceledError as ex_canceled:
        logger.error('Query timed out', None, ex_canceled, redshift_query=str(redshift_query), query_params=str(params), timeout=QUERY_TIMEOUT)
        raise
    except psycopg2.Error as ex_generic:
        logger.error('Error while retrieving data', None, ex_generic, redshift_query=str(redshift_query), query_params=str(params))
        raise
    finally:
        if connection is not None:
            connection.close()
        return result

