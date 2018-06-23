Write-Host "============================================================"
Write-Host "python C128F10.py --k_allTrain_epochs 1"
Write-Host "============================================================"
python C128F10.py --k_allTrain_epochs 1 
Write-Host "============================================================"
Write-Host "python C128F10.py --k_allTrain_epochs 5"
Write-Host "============================================================"
python C128F10.py --k_allTrain_epochs 5 
for($i=0; $i -le 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python C128F10.py --k_allTrain_epochs  $i "
    Write-Host "============================================================"
    python C128F10.py --k_allTrain_epochs $i 
}

