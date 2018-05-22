for($i=64;$i -le 1024;$i=$i*2)
{
    python 1layermicro.py $i 1
    python 1layermicro.py $i 10
    python 1layermicro.py $i 20
}




