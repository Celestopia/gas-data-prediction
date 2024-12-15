# Open Git Bash in the directory of this file, and run ./LR_run_script.sh

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

output_var_names=(
    "烟气含氧量（CEMS） \
    NOX浓度 \
    一氧化碳 \
    烟气湿度（CEMS） \
    烟气压力（CEMS） \
    烟气温度（CEMS） \
    炉膛出口烟气压力")

D:/Anaconda3/envs/env_py312/python.exe _LR_exe.py \
    --model_name LR \
    --data_path "E:\\科创优才\\实验数据\\天然气锅炉数据1.xlsx" \
    --sheet_name "稳定运行数据段" \
    --input_len $input_len \
    --output_len 1 \
    --overlap 1 \
    --input_var_names "$input_var_names" \
    --output_var_names "$output_var_names" \
    --l2_ratio 0.1 \
    --l1_ratio 10.0 \
    --figsize 12:6 \
    --save_directory "./results" \
    --save_record 1 \
    --record_path "./results/LR_record.xlsx"