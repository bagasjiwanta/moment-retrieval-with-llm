# do bump the workers amount if you do not build decord with gpu

python LAVIS/make_dataset.py \
    --prompt_style 1 \
    --dataset_dir /workspace/moment-retrieval-with-llm/datasets \
    --dataset_name qvhighlights \
    --num_frames 10 \
    --num_workers 4 \
    --pretty_json 
