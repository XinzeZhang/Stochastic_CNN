
for($i=110; $i -le 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python SCSF.py --n_kernel 192 --k_allTrain_epochs  $i "
    Write-Host "============================================================"
    python SCSF.py --n_kernel 192 --k_allTrain_epochs $i 
}

