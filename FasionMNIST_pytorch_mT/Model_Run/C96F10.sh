for((i=0; i <= 300; i=i+10))
do
    echo "============================================================"
    echo "python C96F10.py --k_allTrain_epochs " $i ""
    echo "============================================================"
    python C96F10.py --k_allTrain_epochs $i 
done
