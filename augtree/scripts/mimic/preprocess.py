"""
Main module for cleaning mimic data

Some functions copy pasted from https://github.com/huaxiuyao/Wild-Time/blob/main/wildtime/data/mimic.py
Additional reference: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
"""

import logging
import os
import pickle
import random
import re
import sys

import numpy as np
import pandas as pd

from .icd import diagnosis_to_description

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def get_anchor_year(anchor_year_group):
    year_min = int(anchor_year_group[:4])
    year_max = int(anchor_year_group[-4:])
    assert year_max - year_min == 2
    return year_min


def assign_readmission_label(row: pd.Series):
    curr_subject_id = row.subject_id
    curr_admittime = row.admittime

    next_row_subject_id = row.next_row_subject_id
    next_row_admittime = row.next_row_admittime

    if curr_subject_id != next_row_subject_id:
        label = 0
    elif (next_row_admittime - curr_admittime).days > 15:
        label = 0
    else:
        label = 1

    return label


def diag_icd9_to_3digit(icd9: str):
    if icd9.startswith("E"):
        if len(icd9) >= 4:
            return icd9[:4]
        else:
            print(icd9)
            return icd9
    else:
        if len(icd9) >= 3:
            return icd9[:3]
        else:
            print(icd9)
            return icd9


def diag_icd10_to_3digit(icd10: str):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def diag_icd_to_3digit(icd: str):
    if icd[:4] == "ICD9":
        return "ICD9_" + diag_icd9_to_3digit(icd[5:])
    elif icd[:5] == "ICD10":
        return "ICD10_" + diag_icd10_to_3digit(icd[6:])
    else:
        raise


def list_join(lst: str):
    return " <sep> ".join(lst)


def proc_icd9_to_3digit(icd9: str):
    if len(icd9) >= 3:
        return icd9[:3]
    else:
        print(icd9)
        return icd9


def proc_icd10_to_3digit(icd10: str):
    if len(icd10) >= 3:
        return icd10[:3]
    else:
        print(icd10)
        return icd10


def proc_icd_to_3digit(icd: str):
    if icd[:4] == "ICD9":
        return "ICD9_" + proc_icd9_to_3digit(icd[5:])
    elif icd[:5] == "ICD10":
        return "ICD10_" + proc_icd10_to_3digit(icd[6:])
    else:
        raise


def remove_punc(text: str):
    text = re.sub("([,!?:;])", "", text)
    return text


def add_space_to_punc(text: str):
    text = re.sub("([.,!?:;])", r" \1 ", text)
    text = re.sub("\s{2,}", " ", text)
    return text


def remove_brackets(text: str):
    text = re.sub("\[\*\*.*\*\*\]", " ", text).replace("  ", " ")
    return text


def replace_numbers(text: str):
    text = re.sub("[0-9]", "d", text)
    return text


def replace_break(text: str):
    text = re.sub("\\n", " ", text).replace("  ", " ")
    return text


text_cleanup = {
    "replace_numbers": replace_numbers,
    "replace_break": replace_break,
    "remove_brackets": remove_brackets,
    "remove_punc": remove_punc,
    "add_space": add_space_to_punc,
}


def clean_notes(notes: str):
    for p in [
        "replace_numbers",
        "replace_break",
        "remove_brackets",
        "remove_punc",
        "add_space",
    ]:
        new_notes = text_cleanup[p](notes)
    # not all notes have complaint header
    # new_notes = new_notes[new_notes.index('Complaint'):]
    return new_notes


def process_mimic_data(data_dir: str):
    """
    The main function for preprocessing the mimic data. It performs
    the following tasks:
        - loads the hospital data and gets the patient data as well as
        the procedure and diagnoses icd9/10 codes
        - converts the icd9/10 codes to their descriptions
        - loads in the discharge notes and cleans them
        - merges all the data together into one data frame and dumps into
        a pkl

    Implicit folder structure

    /data
        /mimic-iv-2.2
            /hosp
                /patients.csv
                /diagnoses_icd.csv
                /procedures_icd.csv
        /mimic-iv-note-2.2
            /note
                /discharge.csv
        /icdcodes
            /icd{9,10}_{diagnosis, procedure}.{xlsx,txt}


    Args:
        data_dir (str): path to directory
    Returns:
        None
    """

    logging.info("Loading mimic-iv-2.2 hospital data...")
    mimic_hosp_data_dir = os.path.join(data_dir, "mimic-iv-2.2", "hosp")

    for file in ["patients.csv", "diagnoses_icd.csv", "procedures_icd.csv"]:
        if not os.path.isfile(os.path.join(mimic_hosp_data_dir, file)):
            raise ValueError(f"Please download {file} to {mimic_hosp_data_dir}")

    # Patients
    patients = pd.read_csv(os.path.join(mimic_hosp_data_dir, "patients.csv"))
    patients["real_anchor_year"] = patients.anchor_year_group.apply(
        lambda x: get_anchor_year(x)
    )
    patients = patients[
        ["subject_id", "gender", "anchor_age", "anchor_year", "real_anchor_year"]
    ]
    patients = patients.dropna().reset_index(drop=True)

    # Admissions
    admissions = pd.read_csv(os.path.join(mimic_hosp_data_dir, "admissions.csv"))
    admissions["admittime"] = pd.to_datetime(admissions["admittime"]).dt.date
    admissions = admissions.dropna()
    admissions["mortality"] = admissions.hospital_expire_flag
    admissions = admissions.sort_values(by=["subject_id", "hadm_id", "admittime"])
    admissions["next_row_subject_id"] = admissions.subject_id.shift(-1)
    admissions["next_row_admittime"] = admissions.admittime.shift(-1)
    admissions["readmission"] = admissions.apply(
        lambda x: assign_readmission_label(x), axis=1
    )
    admissions = admissions[
        ["subject_id", "hadm_id", "admittime", "mortality", "readmission"]
    ]
    admissions = admissions.dropna().reset_index(drop=True)

    # Diagnoses ICD
    diagnoses_icd = pd.read_csv(os.path.join(mimic_hosp_data_dir, "diagnoses_icd.csv"))
    diagnoses_icd = diagnoses_icd.dropna()
    diagnoses_icd = diagnoses_icd.drop_duplicates()
    diagnoses_icd = diagnoses_icd.sort_values(by=["subject_id", "hadm_id", "seq_num"])
    diagnoses_icd["icd_code"] = diagnoses_icd.apply(
        lambda x: f"ICD{x.icd_version}_{x.icd_code}", axis=1
    )
    diagnoses_icd = (
        diagnoses_icd.groupby(["subject_id", "hadm_id"])
        .agg({"icd_code": list_join})
        .reset_index()
    )
    diagnoses_icd = diagnoses_icd.rename(columns={"icd_code": "diagnoses"})

    # Procedures ICD
    procedures_icd = pd.read_csv(
        os.path.join(mimic_hosp_data_dir, "procedures_icd.csv")
    )
    procedures_icd = procedures_icd.dropna()
    procedures_icd = procedures_icd.drop_duplicates()
    procedures_icd = procedures_icd.sort_values(by=["subject_id", "hadm_id", "seq_num"])
    procedures_icd["icd_code"] = procedures_icd.apply(
        lambda x: f"ICD{x.icd_version}_{x.icd_code}", axis=1
    )
    procedures_icd = (
        procedures_icd.groupby(["subject_id", "hadm_id"])
        .agg({"icd_code": list_join})
        .reset_index()
    )
    procedures_icd = procedures_icd.rename(columns={"icd_code": "procedure"})

    # Merge
    df = admissions.merge(patients, on="subject_id", how="inner")
    df["real_admit_year"] = df.apply(
        lambda x: x.admittime.year - x.anchor_year + x.real_anchor_year, axis=1
    )
    df["age"] = df.apply(
        lambda x: x.admittime.year - x.anchor_year + x.anchor_age, axis=1
    )
    df = df[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "real_admit_year",
            "age",
            "gender",
            "mortality",
            "readmission",
        ]
    ]
    df = df.merge(diagnoses_icd, on=["subject_id", "hadm_id"], how="inner")
    df = df.merge(procedures_icd, on=["subject_id", "hadm_id"], how="inner")
    processed_file = os.path.join(mimic_hosp_data_dir, "processed_mimic_data.csv")
    df.to_csv(processed_file, index=False)

    # Add in icd code desriptions
    logging.info("Convert icd codes...")
    icd_code_data_dir = os.path.join(data_dir, "icdcodes")
    df["diagnoses_long_description"] = df.diagnoses.apply(
        lambda x: diagnosis_to_description(x, icd_code_data_dir, code_type="diagnosis")
    )
    df["procedure_long_description"] = df.procedure.apply(
        lambda x: diagnosis_to_description(x, icd_code_data_dir, code_type="procedure")
    )

    # Add in discharge notes and preprocess them
    logging.info("Clean and load in discharge notes...")
    discharge_df = pd.read_csv(
        os.path.join(data_dir, "mimic-iv-note-2.2", "note", "discharge.csv")
    )
    discharge_df = discharge_df[["hadm_id", "text"]]
    discharge_df.rename(columns={"text": "discharge_notes"}, inplace=True)
    discharge_df.discharge_notes = discharge_df.discharge_notes.apply(clean_notes)
    df = df.merge(discharge_df, how="inner", on="hadm_id")

    res = {
        "mortality": list(df.mortality.values),
        "readmission": list(df.readmission.values),
        "diagnoses": list(df.diagnoses_long_description.values),
        "procedures": list(df.procedure_long_description.values),
        "discharge_notes": list(df.discharge_notes.values),
    }

    with open(os.path.join(data_dir, "mimiciv-2.2.pkl"), "wb") as handle:
        pickle.dump(res, handle)
