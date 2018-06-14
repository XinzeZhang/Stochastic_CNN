python main.py --k_allTrain_epochs 1 --microTrain_epochs 299
python main.py --k_allTrain_epochs 3 --microTrain_epochs 297
for($i=5; $i -le 40; $i=$i+5)
{
    $j=300-$i
    Write-Host "============================================================"
    Write-Host "python main.py --k_allTrain_epochs $i --microTrain_epochs $j"
    Write-Host "============================================================"
    python main.py --k_allTrain_epochs $i --microTrain_epochs $j
}