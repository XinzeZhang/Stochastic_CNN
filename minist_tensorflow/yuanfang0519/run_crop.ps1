python crop_mnist.py 1
python crop_mnist.py 2
python crop_mnist.py 3
python crop_mnist.py 4
python crop_mnist.py 5
$i=1
for($j=1; $j -le 5; $j=$j+1)
{
    Write-Host "============================================================"
    Write-Host "python crop_mnist_micro.py $i $j"
    Write-Host "============================================================"
    python crop_mnist_micro.py $i $j
    Write-Host "============================================================"
    Write-Host "python crop_mnist_micro_lastwolayer.py $i $j"
    Write-Host "============================================================"
    python crop_mnist_micro_lastwolayer.py $i $j
}
