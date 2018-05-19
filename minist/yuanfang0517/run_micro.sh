# python mnist_base_micro.py 1
# python best_micro.py 1
for((i=5;i<=40;i=i+5))
do
	python mnist_base_micro.py $i
	python best_micro.py $i
done
