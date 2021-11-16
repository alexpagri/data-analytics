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


## Data analytics
to be continued