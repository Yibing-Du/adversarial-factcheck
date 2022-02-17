# setup
pretrained='roberta-large'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
out_dir="output"
mkdir -p "${out_dir}"

data_dir='../data'
pred_sent_dir='../sentence-selection/output'

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ $file -nt $latest ]] && latest=$file
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

echo "Latest checkpoint is ${latest}"

split='shared_task_dev'
for corpus in 'baseline' 'advadd' 'oracle'; do # min & full
  if [[ -f "${out_dir}/${split}.jsonl" ]]; then
    echo "Result '${out_dir}/${split}.jsonl' exists!"
    exit
  fi

  python '../../preprocess_claim_verification.py' \
    --corpus "${data_dir}/corpus_${corpus}.jsonl" \
    --in_file "${pred_sent_dir}/${corpus}.jsonl" \
    --out_file "${out_dir}/${corpus}.tsv"

  python '../../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${out_dir}/${corpus}.tsv" \
    --out_file "${out_dir}/${corpus}.out" \
    --batch_size 16 \
    --gpus 1

  python '../../postprocess_claim_verification.py' \
    --data_file "${data_dir}/shared_task_dev.jsonl" \
    --pred_sent_file "${pred_sent_dir}/${corpus}.jsonl" \
    --pred_claim_file "${out_dir}/${corpus}.out" \
    --out_file "${out_dir}/${corpus}.jsonl"

  python '../../eval_fever.py' \
    --gold_file "${data_dir}/shared_task_dev.jsonl" \
    --pred_file "${out_dir}/${corpus}.jsonl" \
    --out_file "${out_dir}/eval.${corpus}.txt"

# advmod
for split in 'add_para_first1' 'add_para_first2' 'add_para_first3' 'add_para_first4' 'add_para_first5' 'add_sent_first1' 'add_sent_first2' 'add_sent_first3' 'add_sent_first4' 'add_sent_first5' 'advmod_oracle'; do
  if [[ -f "${out_dir}/${split}.jsonl" ]]; then
    echo "Result '${out_dir}/${split}.jsonl' exists!"
    exit
  fi

  python '../../advmod_preprocess_claim_verification.py' \
    --corpus "${data_dir}/corpus_baseline.jsonl" \
    --advmod_aux "${pred_sent_dir}/${split}.tsv" \
    --in_file "${pred_sent_dir}/${split}.jsonl" \
    --out_file "${out_dir}/${split}.tsv"

  python '../../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${out_dir}/${split}.tsv" \
    --out_file "${out_dir}/${split}.out" \
    --batch_size 32 \
    --gpus 1

  python '../../postprocess_claim_verification.py' \
    --data_file "${data_dir}/shared_task_dev.jsonl" \
    --pred_sent_file "${pred_sent_dir}/${split}.jsonl" \
    --pred_claim_file "${out_dir}/${split}.out" \
    --out_file "${out_dir}/${split}.jsonl"

  python '../../eval_fever.py' \
    --gold_file "${data_dir}/shared_task_dev.jsonl" \
    --pred_file "${out_dir}/${split}.jsonl" \
    --out_file "${out_dir}/eval.${split}.txt"
done
