python main.py --k_allTrain_epochs 1 --microTrain_epochs 199
python main.py --k_allTrain_epochs 3 --microTrain_epochs 197
for($i=5; $i -le 40; $i=$i+5)
{
    $j=200-$i
    Write-Host "============================================================"
    Write-Host "python main.py --k_allTrain_epochs $i --microTrain_epochs $j"
    Write-Host "============================================================"
    python main.py --k_allTrain_epochs $i --microTrain_epochs $j
}