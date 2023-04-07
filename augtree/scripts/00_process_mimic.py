"""
Implicit folder structure

/data_dir
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
"""

from scripts.mimic.preprocess import process_mimic_data

DATA_DIR = "/root/.data/llmtree/"


def main(data_dir=None):
    process_mimic_data(data_dir=data_dir)


if __name__ == "__main__":
    main(data_dir=DATA_DIR)
