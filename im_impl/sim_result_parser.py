import os


### Simple script to enable the bulk parse of sim-results. Interesting, when obtaining the intersection model betas via gridsearch.
### The paths have to be adapted.

IMPORT_DIRECTORY = 'alex/grid_search_results'

files = []
for r, d, f in os.walk(IMPORT_DIRECTORY, followlinks=True):
    for file in f:
        if '.xml' in file:
            name = file.split('.xml')[0]
            os.system('python ~/GIT/sumo/tools/xml/xml2csv.py %s/%s' % (IMPORT_DIRECTORY, file))
            #os.system('mv %s/%s.csv ../../ma_tests/intersection_model_sims' % (IMPORT_DIRECTORY, name))
print('Done')
