wget -O  ./datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip  ./datasets.zip -d  ./EURLEX57K
rm  ./datasets.zip
rm -rf  ./EURLEX57K/__MACOSX
mv  ./EURLEX57K/dataset/*  ./EURLEX57K/
rm -rf  ./EURLEX57K/dataset
wget -O  ./EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
