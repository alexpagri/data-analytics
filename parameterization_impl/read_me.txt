Anleitung für SUMO-Szenario Evaluation (Parameterization):

1. OSM Ausschnitt wählen (https://tools.geofabrik.de/calc/)
2. python path/to/sumo/tools/osmGet.py --bbox 13.4085,52.4978,13.4288,52.5043 (--bbox aus Schritt 1)
3. python path/to/sumo/tools/osmBuild.py --osm-file XXX.osm.xml -n --bikelanes.guess      
4. Netfile in netedit schön machen. Fahrradwege anpassen, Kreuzungen zusammenlegen (F7+F5) und unnötige Wege und Ampeln entfernen. Szenario Route f. Radfahrer erzeugen.
5. Default und New_params Ordner anlegen in parameterization_evaluation
6. Über website aus 1. Start- und Endbox für Szenario bestimmen. Diese ins entsprechende Jupyter Notebook eintragen ($start_rect, $end_rect).
7. XXX_trips.xml erstellen. Dafür:
	7a. (falls gewünscht. Die Hinzunahme von AutoVerkehr beeinflusst die Simulation nicht gravierend) path/to/sumo/tools/randomTrips.py -o XXX.trips.xml -n XXX.net.xml -e 9998 --validate --min-distance 500 -p 6 (-p gibt die Pause zw. Autotrips ergo Menge des Verkehrs)
	7b. python createVehTypeDistribution_extended.py config.txt. Die entstandene vTypeDistribution in XXX_trips.xml kopieren. 
	7c. <flow.../> aus vorhandenden XXX_trips.xml ziehen. Ggf. ID der Route anpassen.
8. XXX.sumoconfig per Hand erstellen (siehe vorhandene)
9. path/to/sumo/bin/sumo -c XXX.sumocfg --fcd-output fcd_out.xml --device.fcd.explicit vehDist --fcd-output.geo 
10. python path/to/sumo/tools/xml/xml2csv.py fcd_out.xml
11. mv fcd_out.csv parameterization_eval/sim_data/XXX_new_params.csv
12. Alle Schritte ab 7. auch für andere Ansätze durchführen (zB. default_slow_car, new_params_pedestrians, usw...)  
12. Jupyter Notebook laufen lassen und sich die Boxplots anschauen :-)
