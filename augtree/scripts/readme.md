### Processing MIMIC-IV data

#### Preliminaries
[MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) (Medical Information Mart for Intensive Care) is a publicly available dataset that contains de-identified health records of over 400,000 patients who were admitted to intensive care units (ICUs) at the Beth Israel Deaconess Medical Center between 2008 and 2019. The dataset includes comprehensive clinical data such as vital signs, laboratory tests, medications, procedures, and imaging studies.

The dataset is hosted by the Institute for Medical Engineering and Science at the Massachusetts Institute of Technology (MIT) and is available to researchers who meet certain eligibility criteria, including completing a data use agreement and obtaining institutional review board (IRB) approval.

In experiments, we used MIMIC-IV-2.2. In addition to downloading the MIMIC dataset, ICD codes 9/10 need to be downloaded as well. ICD codes (International Classification of Diseases) are a standardized system of alphanumeric codes used to classify and identify diseases, injuries, and other medical conditions. They are used by healthcare providers and insurance companies to describe and document patient diagnoses and treatments.

* ICD9
  * ICD9-cm codes downloaded from [here](https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes)
     (Version 32 Full and Abbreviated Code Titles  – Effective October 1, 2014 (ZIP). After unzipping, file labeled CMS32_DESC_LONG_SHORT_DX.xlsx is diagnosis, CMS32_DESC_LONG_SHORT_SG.xlsx is procedure codes)
* ICD10
  * ICD10-cm codes downloaded from [here](https://www.cms.gov/medicare/icd-10/2023-icd-10-cm)
     (2023 Code Descriptions in Tabular Order - updated 01/11/2023 (ZIP), the file labeled icd10cm_codes_2023.txt is diagnosis, icd10cm_order_2023.txt is procedures)
  * ICD10-pcs codes (procedure codes) downloaded from [here](https://www.cms.gov/medicare/icd-10/2022-icd-10-pcs) (Download 2022 ICD-10-PCS Order File (Long and Abbreviated Titles) - updated December 1, 2021 (ZIP)

After downloading everything your folder structure should be as follows
```
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
```
####

