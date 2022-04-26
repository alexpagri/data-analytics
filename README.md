# ma
Source code to comprehend thesis.

Detailed readme files are included in the respective sub folders.

-----

*Extended and modified by the project group: **A Cyclist Model for SUMO** in the (A)DSP module in WS 21/22.*

This repository has been extended by a docker setup for the PostgreSQL/PostGIS database and functionalities like multiprocessor importing (among others).

## Requirements
Python 3.7.X, Docker (+ Docker Compose), Preferably a Python virtual environment

## Initialization
All commands are meant to be run in the root directory of this repository.

### Install python dependencies:
```pip install -U pip && pip install -U -r requirements.txt```  
(with the respective virtual environment enabled)

### Configure 'nb-clean'
```nb-clean add-filter --remove-empty-cells```  
This configures the `nb-clean` tool which cleans jupyter notebooks before they are staged in git (outputs, cell run count, cell metadata and empty cells will be removed).

### Run docker containers:  
`docker compose up` starts a PostgreSQL/PostGIS container and a Adminer webinterface container. The latter can be reached at `localhost:8080` and can be used to get and overview of the database and to configure it manually.

### Download datasets:
https://github.com/simra-project contains the newest data.
Older data can be found here:
* https://depositonce.tu-berlin.de/handle/11303/11713
* https://depositonce.tu-berlin.de/handle/11303/13664

The complete dataset is only available under a Non-Disclosure Agreement.

## Data import
To import the data you may run e.g.:  
```python db_importer/import.py```  
Before that you need to configure the location of the dataset(s) in the `settings.py` like e.g.:  
```IMPORT_DIRECTORY = "../datasets/"```  
In this case the datasets would be expected in a directory on the same level as this repositories directory.

## Backup and restore database
To dump the data to a TAR backup file, run:
```./scripts/backup_db_to_file.sh BACKUP_NAME```  
where `BACKUP_NAME` is the name the backup file should have (e.g. `Berlin`, no suffix!).

To import the data from a database dump (in TAR format), place the respective backup file in `postgres-data/backup` and run:  
```./scripts/restore_db_from_backup.sh BACKUP_NAME```  
where `BACKUP_NAME` is the name of the backup file (e.g. `Berlin`, no suffix!).

## Data analysis
All data analysis which was performed by the project group can be found in the directory `adsp`.

The main contributions are organised as follows:
* `parameter_analysis`
    * contains Jupyter notebooks for the analysis of the velocity and acceleration parameters
    * distributions are fitted here, their parameters are extracted and the respective plots are created
* `parameter_evaluation`
    * subdirectory `sim_scenarios` contains SUMO files which describe the five meta-scenarios, based on which the simpler other scenarios are created
    * the Juypyter notebook `simple-path_comparison.ipynb` in which those simpler, straight-path scenarios are described and which can be run to generate the scenario specific evaluation plots as well as the JS-divergence plot
    * the script `run_simulations.sh` can be run to run the actual simulations in SUMO and takes as single parameter either `ALL` or the name of the meta-scenario (eg. `B1`)
        * it is expected to be run in a VM as it is set up in this repository: https://github.com/ADSP-Cyclist-Model-for-SUMO/Ubuntu20.04_SUMO_DevEnv
* `turn_analyis`
    * TO BE DONE



