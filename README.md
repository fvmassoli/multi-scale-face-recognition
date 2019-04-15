python main.py -n names.csv -f features.txt -l labels.txt -gpu -url out2.mp4 -v 1 -ht -ksd 0.6 -dc 0.417 -gc -p 8032


## Run model in recognize mode: defualt --> knn, k_neighbours=10
## WARNING: k_neighbours can't be more than the number of pictures stored.
python main.py -aw -sv -n names.csv -f features.txt -l labels.txt

## Run model in recognize mode: use centroid instead of single feature
## WARNING: with the -gc flag, -k (i.e. k_neighbours) can't be more than the number of identities. 
## The other classifiers doesn't care about -k
python main.py -aw -sv -n names.csv -f features.txt -l labels.txt -gc -k #

## Run model in recognize mode: use -clft flag to use a different classfier---WARNING: GausianProcesses and SVM requires more than one class to work
python main.py -aw -sv -n names.csv -f features.txt -l labels.txt

## Run model in training mode: use -nti flag to set set the maximum number of training pictures from which acquire features
## the flag -t sets the type of training
## if you want to acquire form folder pass the path with the flag -d.
## The folder structure has to be like:
## 					root/
##					root/id_1
##					root/id_2 
##					   :
##					root/id_n
python main.py -aw -sv -n names.csv -f features.txt -l labels.txt -atm -t #
# multi-scale-face-recognition
