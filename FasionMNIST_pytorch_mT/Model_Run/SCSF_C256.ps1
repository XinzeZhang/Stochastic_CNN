for($i=100; $i -le 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python SCSF.py --n_kernel 256 --k_allTrain_epochs  $i "
    Write-Host "============================================================"
    python SCSF.py --n_kernel 256 --k_allTrain_epochs $i 
}

