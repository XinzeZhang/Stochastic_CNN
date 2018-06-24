Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 384 --k_allTrain_epochs 1"
Write-Host "============================================================"
python SCSF.py --n_kernel 384 --n_kernel 384 --k_allTrain_epochs 1 
Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 384 --k_allTrain_epochs 5"
Write-Host "============================================================"
python SCSF.py --n_kernel 384 --k_allTrain_epochs 5 
for($i=0; $i -le 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python SCSF.py --n_kernel 384 --k_allTrain_epochs  $i "
    Write-Host "============================================================"
    python SCSF.py --n_kernel 384 --k_allTrain_epochs $i 
}

