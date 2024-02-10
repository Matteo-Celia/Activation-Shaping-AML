exp=${1}
target_domain=${2}


python main.py \
--experiment= ${exp} \
--experiment_name= ${exp}/${target_domain}/ \
--dataset_args="{'root': 'data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}' }" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1 \