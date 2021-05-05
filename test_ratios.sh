#! /bin/sh

source bin/activate

for i in .2 .5 1; do
    if [[ -d "ratio-$i" ]] ; then 
        rm -rf "ratio-$i"
    fi

    mkdir "ratio-$i"
    python make_sub_dataset.py -i 0 2 3 4 5 -d new-dataset.csv -s column_stats.csv -m .1 -o "ratio-$i"/core-dataset.csv -u "$i"
    python find_usage_ratio.py -g "ratio-$i"/core-dataset.csv "ratio-$i"/feature-ratio.csv
    python find_usage_ratio.py -g  -t "ratio-$i"/core-dataset.csv "ratio-$i"/instance-ratio.csv
    python plot_csv.py -a -i "ratio-$i"/feature-ratio.csv -o "ratio-$i"/feature-ratio.png -t "Rapporto di utilizzo delle feature" -l "Rapporto" --x-label 'Indice Feature' --y-label 'Rapporto'
    python plot_csv.py -a -i "ratio-$i"/instance-ratio.csv -o "ratio-$i"/instance-ratio.png -t "Rapporto di utilizzo delle Istanze" -l "Rapporto" --x-label 'Indice Istanza' --y-label 'Rapporto'
    python column_stats.py "ratio-$i"/core-dataset.csv "ratio-$i"/column_stats.csv
done


