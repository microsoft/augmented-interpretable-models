import os
import pickle
import re

import pandas as pd


def save_icd9_diagnosis_procedures(data_dir: str):
    """
    get icd9 diagnosis/procedure codes and descriptions

    (data_dir): directory that contains xlsx files
    """
    if not os.path.exists(os.path.join(data_dir, "icd9_diagnosis.pkl")):
        icd9_diagnosis = pd.read_excel(os.path.join(data_dir, "icd9_diagnosis.xlsx"))
        icd9_diagnosis.columns = [
            "diagnosis_code",
            "long_description",
            "short_description",
        ]
        icd9_diagnosis_dict = dict(
            zip(icd9_diagnosis.diagnosis_code, icd9_diagnosis.long_description)
        )

        with open(os.path.join(data_dir, "icd9_diagnosis.pkl"), "wb") as handle:
            pickle.dump(icd9_diagnosis_dict, handle)

    if not os.path.exists(os.path.join(data_dir, "icd9_procedure.pkl")):
        icd9_procedure = pd.read_excel(os.path.join(data_dir, "icd9_procedure.xlsx"))
        icd9_procedure.columns = [
            "procedure_code",
            "long_description",
            "short_description",
        ]
        icd9_procedure.procedure_code = icd9_procedure.procedure_code.astype(str)
        icd9_procedure_dict = dict(
            zip(icd9_procedure.procedure_code, icd9_procedure.long_description)
        )

        with open(os.path.join(data_dir, "icd9_procedure.pkl"), "wb") as handle:
            pickle.dump(icd9_procedure_dict, handle)


def save_icd10_diagnosis_procedures(data_dir: str):
    """
    get icd9 diagnosis/procedure codes and descriptions

    (data_dir): directory that contains xlsx files
    """

    if not os.path.exists(os.path.join(data_dir, "icd10_diagnosis.pkl")):
        icd10_diagnosis_dict = {}
        with open(os.path.join(data_dir, "icd10_diagnosis.txt"), "r") as file:
            lines = file.readlines()
            for line in lines:
                # only up to the first 8 characters in the line are the codes
                code = line[:8].strip(" ")
                description = line[8:].strip(" ").strip("\n")
                icd10_diagnosis_dict[code] = description

        with open(os.path.join(data_dir, "icd10_diagnosis.pkl"), "wb") as handle:
            pickle.dump(icd10_diagnosis_dict, handle)

    if not os.path.exists(os.path.join(data_dir, "icd10_procedure.pkl")):
        icd10_procedure_dict = {}
        with open(os.path.join(data_dir, "icd10_procedure.txt"), "r") as file:
            lines = file.readlines()
            for line in lines:
                ## HACK
                # there is a strict separation between descrpitions after 77 characters
                # so manually split the line there and add whitespace so that the regex works
                new_line = line[:77] + "         " + line[77:]
                match = re.search(
                    "^\d{5}\s+([A-Za-z0-9]+)\s+[01]{1}\s+.+\s+(\s+.+)", new_line
                )
                code = match.group(1).strip(" ")
                description = match.group(2).strip(" ").strip("\n")
                icd10_procedure_dict[code] = description

        with open(os.path.join(data_dir, "icd10_procedure.pkl"), "wb") as handle:
            pickle.dump(icd10_procedure_dict, handle)


def diagnosis_to_description(codes, data_dir=None, code_type="diagnosis"):
    """
    convert list of diagnoses to descriptions

    codes (str): e.g 'ICD9_123 <sep> ICD10_1231 <sep> ICD9_4432'
    """
    descriptions = []
    untracked_codes = set()
    assert (
        code_type == "diagnosis" or code_type == "procedure"
    ), f"code type {code_type} not understood"

    # save the pickles if they dont already exist
    if not os.path.exists(os.path.join((data_dir), f"icd9_{code_type}.pkl")):
        save_icd9_diagnosis_procedures(data_dir)
    if not os.path.exists(os.path.join((data_dir), f"icd10_{code_type}.pkl")):
        save_icd10_diagnosis_procedures(data_dir)

    # load in the appropriate dictionaries
    with open(os.path.join(data_dir, f"icd9_{code_type}.pkl"), "rb") as handle:
        icd9_dict = pickle.load(handle)
    with open(os.path.join(data_dir, f"icd10_{code_type}.pkl"), "rb") as handle:
        icd10_dict = pickle.load(handle)

    for icd_code in codes.split(" <sep> "):
        code_type = re.search("ICD([0-9]+)_", icd_code).group(1)
        code_number = re.search("_(.*)", icd_code).group(1)
        if int(code_type) == 9:
            try:
                description = icd9_dict[code_number]
            except:
                try:
                    new_code_number = code_number + "0"
                    description = icd9_dict[new_code_number]
                except:
                    try:
                        new_code_number = code_number + "1"
                        description = icd9_dict[new_code_number]
                    except:
                        untracked_codes.add(icd_code)
                        continue
        elif int(code_type) == 10:
            try:
                description = icd10_dict[code_number]
            except:
                try:
                    new_code_number = code_number + "0"
                    description = icd10_dict[new_code_number]
                except:
                    try:
                        new_code_number = code_number + "1"
                        description = icd10_dict[new_code_number]
                    except:
                        untracked_codes.add(icd_code)
                        continue
        else:
            print(icd_code)
            print(code_type, code_number)
            raise ValueError("code type not understood")
        descriptions.append(description)

    return descriptions
