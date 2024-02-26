#/bin/bash
# check the enviroment and get the sumbmitter information,
set -e
#nvidia-smi
cd /output
# DO NOT EDIT THIS LINE
python /before_run.py

# You can add more commands here to init your enviromentt
# Run your inference code and output the result to /output
python /test_submission.py

# DO NOT EDIT THIS LINE
python /after_run.py