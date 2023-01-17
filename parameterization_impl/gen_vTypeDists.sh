#!/bin/bash

python createVehTypeDistribution_extended.py -o vTypeDistributions_new_params_all.add.xml config_all.txt
python createVehTypeDistribution_extended.py -o vTypeDistributions_new_params_fast.add.xml config_fast.txt
python createVehTypeDistribution_extended.py -o vTypeDistributions_new_params_medium.add.xml config_medium.txt
python createVehTypeDistribution_extended.py -o vTypeDistributions_new_params_slow.add.xml config_slow.txt