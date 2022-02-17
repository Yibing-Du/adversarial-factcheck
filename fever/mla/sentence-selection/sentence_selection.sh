# setup
pretrained='bert-base-uncased'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
data_dir='../data'
out_dir="output"
mkdir -p "${out_dir}"

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ $file -nt $latest ]] && latest=$file
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

echo "Latest checkpoint is ${latest}"

# original + advadd
split='shared_task_dev'
for corpus in 'baseline' 'advadd' 'oracle'; do
  if [[ -f "${out_dir}/${split}.jsonl" ]]; then
    echo "${out_dir}/${split}.jsonl exists!"
    continue
  fi

  python '../preprocess_sentence_selection.py' \
    --corpus "${data_dir}/corpus_${corpus}.jsonl" \
    --in_file "${data_dir}/${corpus}_${split}.jsonl" \
    --out_file "${out_dir}/${corpus}.tsv"

  python '../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${out_dir}/${corpus}.tsv" \
    --out_file "${out_dir}/${corpus}.out" \
    --batch_size 256 \
    --gpus 1

  python '../postprocess_sentence_selection.py' \
    --in_file "${out_dir}/${corpus}.tsv" \
    --pred_sent_file "${out_dir}/${corpus}.out" \
    --pred_doc_file "${data_dir}/${split}.jsonl" \
    --out_file "${out_dir}/${corpus}.jsonl" \
    --max_evidence_per_claim 5

  python '../eval_sentence_selection.py' \
    --gold_file "${data_dir}/${split}.jsonl" \
    --pred_file "${out_dir}/${corpus}.jsonl" \
    --out_file "${out_dir}/eval.${corpus}.txt"
done

# advmod
for split in 'add_para_first1' 'add_para_first2' 'add_para_first3' 'add_para_first4' 'add_para_first5' 'add_sent_first1' 'add_sent_first2' 'add_sent_first3' 'add_sent_first4' 'add_sent_first5' 'advmod_oracle'; do
  if [[ -f "${out_dir}/${split}.jsonl" ]]; then
    echo "${out_dir}/${split}.jsonl exists!"
    continue
  fi

  python '../preprocess_sentence_selection.py' \
    --corpus "${data_dir}/original_corpus.jsonl" \
    --in_file "${data_dir}/document-retrieval/original/${split}.jsonl" \
    --out_file "${out_dir}/${split}.tsv"

  python '../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${out_dir}/${split}.tsv" \
    --out_file "${out_dir}/${split}.out" \
    --batch_size 256 \
    --gpus 1

  python '../postprocess_sentence_selection.py' \
    --in_file "${out_dir}/${split}.tsv" \
    --pred_sent_file "${out_dir}/${split}.out" \
    --pred_doc_file "${data_dir}/document-retrieval/shared_task_dev_advmod.jsonl" \
    --out_file "${out_dir}/${split}.jsonl" \
    --max_evidence_per_claim 5

  python '../eval_sentence_selection.py' \
    --gold_file "${data_dir}/shared_task_dev.jsonl" \
    --pred_file "${out_dir}/${split}.jsonl" \
    --out_file "${out_dir}/eval.${split}.txt"
done
