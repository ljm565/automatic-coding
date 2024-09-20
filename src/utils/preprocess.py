import sys
sys.path.append('/Users/ljm/Documents/Python/automatic-coding/src')

import pandas as pd
from tqdm import tqdm
from typing import List
from functools import reduce

from utils import LOGGER, colorstr
from utils.filesys_utils import csv_read



def get_code_dict(paths: List[str], columns: List[str]) -> dict:
    """This function makes ICD code labels.

    Args:
        paths (List[str]): List of ICD code data file paths.
        columns (List[str]): List of corresponding column names in the data files.
    
    Returns:
        dict: Dictionary mapping unique ICD codes to indices.
    """
    if len(paths) != len(columns):
        raise AssertionError(colorstr('red', f'Length of paths and columns must be equal, but got {len(paths)} and {len(columns)}.'))

    code_dict = dict()
    for data_id, (path, col) in enumerate(zip(tqdm(paths, desc='Pre-processing ICD codes..'), columns)):
        codes = csv_read(path)[col].tolist()
        assert len(codes) == len(set(codes)), colorstr("red", f"Duplicate ICD code '{code}' found in '{path}'. Codes must be unique.")
        
        for code in codes:
            code = code + f'.__{data_id}__'
            code_dict[code] = len(code_dict)
    
    LOGGER.info(f'Pre-processing ICD codes done. Total unique codes: {len(code_dict)}')
    return code_dict



def get_hadm_dict(hadm_data_paths: List[str], code_data_paths: List[str], columns: List[str]) -> dict:
    """This function maps HADM_ID to each their label.

    Args:
        hadm_data_paths (List[str]): List of HADM data file paths.
        code_data_paths (List[str]): List of ICD code data file paths.
        columns (List[str]): List of corresponding column names in the code data files.

    Returns:
        dict: Dictionary mapping HADM to ICD code list.
    """
    LOGGER.info(f'Pre-processing HADM data..')
    if len(hadm_data_paths) != len(code_data_paths):
        raise AssertionError(colorstr('red', f'Length of paths and columns must be equal, but got {len(hadm_data_paths)} and {len(code_data_paths)}.'))

    code_dict = get_code_dict(code_data_paths, columns)
    hadm_dict = dict()
    error_codes = {i: [] for i in range(len(hadm_data_paths))}

    for i in tqdm(range(len(hadm_data_paths)), desc='Pre-processing HADM and ICD codes..'):
        hadm_data = csv_read(hadm_data_paths[i])
        hadms, icd_codes = hadm_data['HADM_ID'].tolist(), hadm_data[columns[i]].tolist()
        
        for hadm, icd_code in zip(hadms, icd_codes):
            try:
                if hadm in hadm_dict:
                    hadm_dict[hadm].append(code_dict[f'{icd_code}.__{i}__'])
                else:
                    hadm_dict[hadm] = [code_dict[f'{icd_code}.__{i}__']]
            except KeyError:    # Except mismatch cases of ICD codes
                if icd_code not in error_codes[i]:
                    LOGGER.warning(colorstr("yellow", f"'{icd_code}' is not in the {hadm_data_paths[i]} data."))
                    error_codes[i].append(icd_code)

    LOGGER.info(f'Pre-processing HADM data done. Total HADM data: {len(hadm_dict)}')
    return hadm_dict


def get_noteevent_dict(path:str, chunk_size:int=None) -> dict:
    """This function maps HADM_ID to NOTEEVENTS texts.

    Args:
        path (str): NOTEEVENTS.csv data path.
        chunk_size (int, optional): Pandas reading chunk size. Defaults to 100000. Defaults to None.

    Returns:
        dict: Dictionary mapping HADM to NOTEEVENTS text list.
    """
    LOGGER.info(f'Pre-processing NOTEEVENT data..')

    # Filtering conditions
    # NOTE: You can add other conditions in the below.
    cond1 = lambda df: df['CATEGORY'] == 'Discharge summary'
    cond2 = lambda df: df['DESCRIPTION'] == 'Report'
    cond3 = lambda df: df['HADM_ID'].isnull() == False
    note_events = preprocess_noteevents(path, chunk_size, *[cond1, cond2, cond3])

    # Make note event dictionary
    noteevent_dict = dict()
    for hadm, text in zip(tqdm(note_events.iloc[:, 0].tolist(), desc='Pre-processing HADM and NOTEEVENT texts..'), note_events.iloc[:, 1].tolist()):
        if hadm in noteevent_dict:
            noteevent_dict[hadm].append(text)
        else:
            noteevent_dict[hadm] = [text]
    
    LOGGER.info(f'Pre-processing NOTEEVENTS data done. Total HADM and NOTEEVENTS data: {len(noteevent_dict)}')
    return noteevent_dict


def preprocess_noteevents(path:str, chunk_size:int=100000, *conditions:List) -> pd.DataFrame:
    """NOTEEVENTS.csv pre-processing function.

    Args:
        path (str): NOTEEVENTS.csv data path.
        chunk_size (int, optional): Pandas reading chunk size. Defaults to 100000.

    Returns:
        pd.DataFrame: Concatenated Pandas dataframe.
    """
    df = pd.concat(list(csv_read(path, chunk_size))).loc[:, ['HADM_ID', 'CATEGORY', 'DESCRIPTION', 'TEXT']]  # get only necessary columns
    
    # filtering
    conditions = [condition(df) for condition in conditions]
    all_and_condition = reduce(lambda x, y: x & y, conditions)  # all and condition
    df = df[all_and_condition].loc[:, ['HADM_ID', 'TEXT']]
    
    return df