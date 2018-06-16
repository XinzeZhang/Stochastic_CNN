python unit_test.py --k_allTrain_epochs 1 
python unit_test.py --k_allTrain_epochs 5 
for($i=0; $i -lt 250; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python unit_test.py --k_allTrain_epochs $i "
    Write-Host "============================================================"
    python unit_test.py --k_allTrain_epochs $i 
}
for($i=250; $i -lt 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python unit_test.py --k_allTrain_epochs $i "
    Write-Host "============================================================"
    python unit_test.py --k_allTrain_epochs $i 
}