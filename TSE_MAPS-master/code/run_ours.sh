mode=gpt_ours
model_name=gpt-3.5-turbo-0125 # Please select from gpt-3.5-turbo-0125, meta-llama/Meta-Llama-3.1-70B-Instruct and Qwen/Qwen2-72B-Instruct
output_base_dir=/data/ubuntu/ut_gen/output/
mkdir -p $output_base_dir/$mode/train

python auto_prompt_ours.py --seed_number 5 --iteration_number 4 --max_test_cases 5 --generated_number 2 \
                                --mode $mode --model_name $model_name --output_base_dir $output_base_dir \
                                --data_dir /data/ubuntu/ut_gen/data/sample_ours.jsonl \
                                --test_data_dir /data/ubuntu/ut_gen/data/all_ours.jsonl \
                                --seed_prompt_addr /data/ubuntu/ut_gen/code/seed_prompt.txt \
# the iteration_number contains the first seed results, sothe value is 4