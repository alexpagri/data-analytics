#!/bin/bash

# ssh vagrant@localhost -p 2222 -i ../SUMO/Ubuntu20.04_SUMO_DevEnv/.vagrant/machines/SUMO_Dev/virtualbox/private_key

SCENARIO_FOLDER="sim_scenarios"
SIM_DATA_FOLDER="sim_data"

SCENARIO_NAME_SUFFIXES=("default" "new_params_all" "new_params_slow" "new_params_medium" "new_params_fast")

run_simulation() {
    # mkdir tmp_sim

    if [[ -e $SCENARIO_FOLDER/$SCENARIO_SUB_FOLDER/custom_seconds.txt ]]
    then
        local SECONDS=$(cat $SCENARIO_FOLDER/$SCENARIO_SUB_FOLDER/custom_seconds.txt)
    else
        local SECONDS=10000
    fi

    cat $SCENARIO_FOLDER/template.sumocfg | \
        sed -e "s/#scenario#/$SCENARIO_SUB_FOLDER/g" | \
        sed -e "s/#seconds#/$SECONDS/g" | \
        sed -e "s/#sub#/..\/..\/..\/..\/parameterization_impl\/vTypeDistributions_$SUFFIX.add.xml/g" > \
        $SCENARIO_FOLDER/$SCENARIO_SUB_FOLDER/$SCENARIO_NAME.sumocfg

    /bin/sumo -c "$SCENARIO_FOLDER"/"$SCENARIO_SUB_FOLDER"/"$SCENARIO_NAME".sumocfg \
        --fcd-output tmp_sim/"$SCENARIO_NAME".xml --device.fcd.explicit vehDist --fcd-output.geo 

    python3 /usr/share/sumo/tools/xml/xml2csv.py tmp_sim/"$SCENARIO_NAME".xml

    mv tmp_sim/"$SCENARIO_NAME".csv "$SIM_DATA_FOLDER"/"$SCENARIO_NAME".csv

    # rm -rf tmp_sim
}

rm -rf tmp_sim
mkdir tmp_sim

if [[ "$1" == "ALL" ]]; then
    for SUB_FOLDER in $SCENARIO_FOLDER/*/ ; do
        for SUFFIX in "${SCENARIO_NAME_SUFFIXES[@]}"; do
            TMP=${SUB_FOLDER%/} 
            SCENARIO_SUB_FOLDER="${TMP##*/}"
            SCENARIO_NAME="$SCENARIO_SUB_FOLDER"_"$SUFFIX"
            echo "Running scenario "$SCENARIO_NAME"..."
            run_simulation &
        done
    done
else
    SCENARIO_SUB_FOLDER="$1"
    SCENARIO_NAME_SUFFIX="$2"
    SCENARIO_NAME="$SCENARIO_SUB_FOLDER"_"$SCENARIO_NAME_SUFFIX"
    echo "Running scenario "$SCENARIO_NAME"..."
    run_simulation
fi