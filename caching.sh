## RCV1v2
python src/dataset/caching.py data=RCV1v2 stage=TRAIN num_workers=2
python src/dataset/caching.py data=RCV1v2 stage=VAL num_workers=1
python src/dataset/caching.py data=RCV1v2 stage=TEST num_workers=5

## NYT
python src/dataset/caching.py data=NYT stage=TRAIN num_workers=2
python src/dataset/caching.py data=NYT stage=VAL num_workers=1
python src/dataset/caching.py data=NYT stage=TEST num_workers=1

## EURLEX57K
python src/dataset/caching.py data=EURLEX57K stage=TRAIN num_workers=4
python src/dataset/caching.py data=EURLEX57K stage=VAL num_workers=1
python src/dataset/caching.py data=EURLEX57K stage=TEST num_workers=1