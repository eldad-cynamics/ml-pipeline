import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy
import os, sys, glob, re, json, argparse, yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from datetime import datetime, timedelta
from time import time
from loguru import logger

warnings.filterwarnings('ignore')

def get_current_date_and_time()->str:
    return datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

def main ():
    logger.info(f"Running:---{Path(__file__).name}---")


if __name__ == '__main__':
    main()
