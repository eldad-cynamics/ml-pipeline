import functools
import os
import json
import boto3
import base64

from . import logger

config_env = os.environ["CYNAMICS_ENV"]




@functools.lru_cache
@functools.lru_cache(128)
def get_param(key):
    try:
        ssm_client = boto3.client("ssm")
        # if got the param needed as environment param
        if key in os.environ and os.environ[key]:
            return os.environ[key]
        # else, need to fetch it from ssm
        response = ssm_client.get_parameter(Name=f"{config_env}_{key}", WithDecryption=True)
        print(f"got parameter {key} from SSM")
        return response["Parameter"]["Value"]
    except Exception as ex:
        logger.error("Key does not exist in environment / SSM", None, ex, key=key, env=config_env)
        raise


class SSMParamAccessor:
    def __init__(self):
        self.ip_group_discovery_bucket = None

    def get_ip_group_discovery_bucket(self):
        if not self.ip_group_discovery_bucket:
            self.ip_group_discovery_bucket = get_param("IP_GROUP_DISCOVERY_BUCKET")
        return self.ip_group_discovery_bucket


def get_secret(secret_id: str):

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager")
    secret = None

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_id)
    except Exception as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            logger.error("DSecrets Manager can't decrypt the protected secret text using the provided KMS key", None, None)
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            logger.error("An error occurred on the server side", None, None)
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            logger.error("Provided an invalid value for a parameter", None, None)
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            logger.error("Provided a parameter value that is not valid for the current state of the resource", None, None)
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            logger.error("Can't find the resource", None, None)
            raise e

    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        print(f"got secret {secret_id} from Secret Manger")
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
            return json.loads(secret)
        else:
            secret = base64.b64decode(get_secret_value_response["SecretBinary"])
            return secret
