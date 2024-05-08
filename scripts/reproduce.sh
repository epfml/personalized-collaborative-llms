#!/bin/bash

trusts=(none naive static dynamic dynamic-ref dynamic-token)
datasets_name=(agnews_mixed agnews_specific three_multi_mixed three_multi_specific github_wiki_mixed github_wiki_specific)
num_clients=(4 8 9 9 8 8)
project_name=(cl-AG-M cl-AG-S cl-TW-M cl-TW-S cl-GW-M cl-GW-S)

len=6

if [ "$#" -ne 1 ]; then
    runs=10
else
    runs=$1
fi

echo "Starting runs, using $runs runs per experiment"

for trust in "${trusts[@]}"; do
    for ((i = 0; i < len; i++)); do
        dataset_name=${datasets_name[i]}
        num_client=${num_clients[i]}
        proj_name=${project_name[i]}
        echo "---------------- New experiment: $dataset_name, $trust, $num_client, $proj_name ----------------"
        ./scripts/script.sh "$proj_name" "$dataset_name" "$trust" "$num_client" $runs
    done
done