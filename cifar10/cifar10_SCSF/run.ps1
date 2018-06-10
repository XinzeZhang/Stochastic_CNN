for($i=16; $i -le 512; $i=$i*2)
{
    python cifar_10_SCSF.py $i 1
	python cifar_10_SCSF.py $i 0
}

