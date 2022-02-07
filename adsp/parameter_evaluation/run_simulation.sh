#! /usr/bin/env bash

SCENARIO_SUB_FOLDER="$1"
SCENARIO_NAME="$2"

SCENARIO_FOLDER="sim_scenarios"
SIM_DATA_FOLDER="sim_data"

mkdir tmp_sim

/sumo/bin/sumo -c "$SCENARIO_FOLDER"/"$SCENARIO_SUB_FOLDER"/"$SCENARIO_NAME".sumocfg \
    --fcd-output tmp_sim/fcd_out.xml --device.fcd.explicit vehDist --fcd-output.geo 

python /sumo/tools/xml/xml2csv.py tmp_sim/fcd_out.xml

mv tmp_sim/fcd_out.csv "$SIM_DATA_FOLDER"/"$SCENARIO_NAME".csv

rm -rf tmp_sim