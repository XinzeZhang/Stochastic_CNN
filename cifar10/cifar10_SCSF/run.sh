for((i=16; i<=512; i=i*2 ))
do
	python cifar_10_SCSF.py $i 1
	python cifar_10_SCSF.py $i 0
done
