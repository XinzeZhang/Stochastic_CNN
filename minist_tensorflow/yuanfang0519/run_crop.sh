
for((i=5;i<=40;i=i+5))
do
    for((j=1;j<=5;j=j+1))
    do
	    python crop_mnist_micro.py $i $j
	    python crop_mnist_micro_lastwolayer.py $i $j
	done
done
