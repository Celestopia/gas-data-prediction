# Open Git Bash in the directory of this file, and run ./NN_run_script.sh

input_var_names=(
    "主蒸汽流量计算值 \
    锅炉天然气进气流量 \
    锅炉天然气进气温度 \
    锅炉天然气进气压力 \
    鼓风机出口温度 \
    鼓风机出口压力 \
    鼓风机变频器输出反馈 \
    鼓风机变频器电流反馈 \
    冷凝器出口烟气调节阀反馈 \
    SWY大气压 \
    SWY天气温度 \
    SWY空气湿度 \
    SWY湿球温度 \
    主蒸汽温度（蒸汽集箱出口温度） \
    主蒸汽压力（蒸汽集箱出口压力）")

output_var_names=(NOX浓度)

for input_len in {1..10}; do
    echo -e "\n\n\n"
    D:/Anaconda3/envs/env_py312/python.exe _NN_exe.py \
        --model_name MLP \
        --data_path "E:\\科创优才\\实验数据\\天然气锅炉数据1.xlsx" \
        --sheet_name "稳定运行数据段" \
        --input_len $input_len \
        --output_len 1 \
        --overlap 1 \
        --input_var_names "$input_var_names" \
        --output_var_names "$output_var_names" \
        --learning_rate 0.001 \
        --batch_size 128 \
        --num_epochs 200 \
        --optimizer_name "Adam" \
        --device "cuda" \
        --figsize 12:6 \
        --save_directory "./results" \
        --save_record 1 \
        --record_path "./results/NN_record.xlsx"
done


for input_len in {8..12}; do
    echo -e "\n\n\n"
    D:/Anaconda3/envs/env_py312/python.exe _NN_exe.py \
        --model_name CNN \
        --data_path "E:\\科创优才\\实验数据\\天然气锅炉数据1.xlsx" \
        --sheet_name "稳定运行数据段" \
        --input_len $input_len \
        --output_len 1 \
        --overlap 1 \
        --input_var_names "$input_var_names" \
        --output_var_names "$output_var_names" \
        --learning_rate 0.001 \
        --batch_size 32 \
        --num_epochs 200 \
        --optimizer_name "Adam" \
        --device "cuda" \
        --figsize 12:6 \
        --save_directory "./results" \
        --save_record 1 \
        --record_path "./results/NN_record.xlsx"
done



for input_len in {5..10}; do
    echo -e "\n\n\n"
    D:/Anaconda3/envs/env_py312/python.exe _NN_exe.py \
        --model_name LSTM \
        --data_path "E:\\科创优才\\实验数据\\天然气锅炉数据1.xlsx" \
        --sheet_name "稳定运行数据段" \
        --input_len $input_len \
        --output_len 1 \
        --overlap 1 \
        --input_var_names "$input_var_names" \
        --output_var_names "$output_var_names" \
        --learning_rate 0.001 \
        --batch_size 32 \
        --num_epochs 200 \
        --optimizer_name "Adam" \
        --device "cuda" \
        --figsize 12:6 \
        --save_directory "./results" \
        --save_record 1 \
        --record_path "./results/NN_record.xlsx"
done



for input_len in {8..12}; do
    echo -e "\n\n\n"
    D:/Anaconda3/envs/env_py312/python.exe _NN_exe.py \
        --model_name TCN \
        --data_path "E:\\科创优才\\实验数据\\天然气锅炉数据1.xlsx" \
        --sheet_name "稳定运行数据段" \
        --input_len $input_len \
        --output_len 1 \
        --overlap 1 \
        --input_var_names "$input_var_names" \
        --output_var_names "$output_var_names" \
        --learning_rate 0.001 \
        --batch_size 32 \
        --num_epochs 200 \
        --optimizer_name "Adam" \
        --device "cuda" \
        --figsize 12:6 \
        --save_directory "./results" \
        --save_record 1 \
        --record_path "./results/NN_record.xlsx"
done