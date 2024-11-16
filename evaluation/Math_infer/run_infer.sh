models=(/path/to/your/model)
for (( i=0; i<1; i++ )); do
    CUDA_VISIBLE_DEVICES=$i python3 math_infer.py \
    --model ${models[i]} \
    --data_file ./../data/MATH_test.jsonl \
    --tensor_parallel_size 1 \
    >> results_${i}.txt &
done
wait