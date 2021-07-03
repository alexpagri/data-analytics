import os


IMPORT_DIRECTORY = 'sim_results'

files = []
for r, d, f in os.walk(IMPORT_DIRECTORY, followlinks=True):
    for file in f:
        if '.xml' in file:
            name = file.split('.xml')[0]
            os.system('python ../tools/xml/xml2csv.py %s/%s' % (IMPORT_DIRECTORY, file))
            os.system('mv %s/%s.csv ../../ma_tests/intersection_model_sims' % (IMPORT_DIRECTORY, name))
print('Done')
