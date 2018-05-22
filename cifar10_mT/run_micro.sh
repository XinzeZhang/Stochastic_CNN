for((i=16;i<=1024;i=i*2))
do
	python 1layermicro.py $i 1
	python 1layermicro.py $i 10
	python 1layermicro.py $i 20
done

