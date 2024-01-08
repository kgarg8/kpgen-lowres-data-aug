FOLDER=LDKP3km-lr-1000
cd $FOLDER
python preprocess_stage2_v2_sr_aug.py
cd ..
FOLDER=LDKP3km-lr-2000
cd $FOLDER
python preprocess_stage2_v2_sr_aug.py
cd ..
FOLDER=LDKP3km-lr-4000
cd $FOLDER
python preprocess_stage2_v2_sr_aug.py
cd ..
FOLDER=LDKP3km-lr-8000
cd $FOLDER
python preprocess_stage2_v2_sr_aug.py
cd ..