#!/bin/bash

import mne

# run - task
# 1	        Baseline, eyes open
# 2	        Baseline, eyes closed
# 3, 7, 11	Motor execution: left vs right hand
# 4, 8, 12	Motor imagery: left vs right hand

subs = [1,2,3]
runs = [1, 2, 3, 4, 7, 8, 11, 12]
path = "."
sets = mne.datasets.eegbci.load_data(subject=subs, runs=runs, path=path)
[print(filename) for filename in sets]
