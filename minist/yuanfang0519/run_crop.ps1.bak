for($i=5; $i -le 40; $i=$i+5)
{
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
}