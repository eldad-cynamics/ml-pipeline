import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
import os, sys, glob, re, json, argparse, yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta, date
from time import time
import boto3
from loguru import logger

warnings.filterwarnings('ignore')


def stop_sagemaker_notebook_instance (notebook_name: str) -> None:
    sm = boto3.client('sagemaker')
    sm.stop_notebook_instance(NotebookInstanceName=f"{notebook_name}")
