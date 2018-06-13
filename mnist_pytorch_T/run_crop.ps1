for($i=5; $i -le 40; $i=$i+5)
{
    $j=300-$i
    Write-Host "============================================================"
    Write-Host "python main.py --k_allTrain_epochs $i --microTrain_epochs $j"
    Write-Host "============================================================"
    python main.py --k_allTrain_epochs $i --microTrain_epochs $j
}