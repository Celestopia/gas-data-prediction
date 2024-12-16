import sys
import os

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Initialize the settings ------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# --- Dataset settings --------------------------------------------------------------
data_path = r"E:\科创优才\实验数据\天然气锅炉数据1.xlsx"
sheet_name = "1-2月份数据"



input_var_names_list=[
                "主蒸汽流量计算值",
                "锅炉天然气进气流量",
                "锅炉天然气进气温度",
                "锅炉天然气进气压力",
                '鼓风机出口温度',
                "鼓风机出口压力",
                "鼓风机变频器输出反馈",
                "鼓风机变频器电流反馈",
                "冷凝器出口烟气调节阀反馈",
                "SWY大气压",
                "SWY天气温度",
                "SWY空气湿度",
                'SWY湿球温度',
                "主蒸汽温度（蒸汽集箱出口温度）",
                "主蒸汽压力（蒸汽集箱出口压力）",
                ]


input_var_names=' '.join(input_var_names_list)
output_var_names_list=[
                "烟气含氧量（CEMS）",
                #'NO浓度",
                #"NO2浓度", # 主要预测NO，因为NO2的准确性有待考量
                "NOX标干浓度",
                "烟气湿度（CEMS）",
                "烟气压力（CEMS）",
                "烟气温度（CEMS）",
                "一氧化碳",
                #"炉膛出口烟气压力",
                ]
output_var_names=' '.join(output_var_names_list)
input_len = 3
output_len = 1
overlap = 1

# --- Model settings ----------------------------------------------------
model_name = "SVR"

# SVR settings
C=1.0
epsilon=0
nu=0.5
kernel="linear"
degree=3

# LR settings
l2_ratio=0.1
l1_ratio=0.1

# --- Training settings ---------------------------------------------
seed=2024+1216

# General settings
device="cpu"

# SVR settings
tol=0.001
max_iter=-1

# NN settings
optimizer_name="Adam"
learning_rate=0.001
batch_size=32
num_epochs=100

# --- Saving Settings ----------------------------------------------------
save_record=1
figsize="18 12"
record_directory="./_results"

save_log=1
log_path="./_results/_SVR_log.xlsx"
metadata=''


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Set the command --------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def run():
    command=r"python main.py"

    # --- Dataset settings --------------------------------------------------------------
    command+=f' --data_path "{data_path}"'
    command+=f' --sheet_name "{sheet_name}"'
    command+=f' --input_var_names "{input_var_names}"'
    command+=f' --output_var_names "{output_var_names}"'
    command+=f' --input_len {input_len}'
    command+=f' --output_len {output_len}'
    command+=f' --overlap {overlap}'

    # --- Model settings ----------------------------------------------------
    command+=f' --model_name "{model_name}"'

    # SVR settings
    command+=f' --C {C}'
    command+=f' --epsilon {epsilon}'
    command+=f' --nu {nu}'
    command+=f' --kernel "{kernel}"'
    command+=f' --degree {degree}'

    command+=f' --l2_ratio {l2_ratio}'
    command+=f' --l1_ratio {l1_ratio}'

    # --- Training settings ---------------------------------------------------
    # General settings
    command+=f' --seed {seed}'
    command+=f' --device {device}'

    # SVR settings
    command+=f' --tol {tol}'
    command+=f' --max_iter {max_iter}'

    # NN settings
    command+=f' --optimizer_name {optimizer_name}'
    command+=f' --learning_rate {learning_rate}'
    command+=f' --batch_size {batch_size}'
    command+=f' --num_epochs {num_epochs}'

    # --- Saving Settings -------------------------------------------------------
    command+=f' --figsize "{figsize}"'
    command+=f' --save_record {save_record}'
    command+=f' --record_directory "{record_directory}"'
    command+=f' --save_log {save_log}'
    command+=f' --log_path "{log_path}"'
    command+=f' --metadata "{metadata}"'


    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --- Run the command ----------------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("\nRunning command: ", command)
    os.system(command)

run()

'''
import itertools

for i, names in enumerate(itertools.combinations(input_var_names_list, 13)):
    input_var_names=' '.join(names)
    metadata=set(input_var_names_list)-set(names)
    metadata='Discarded variables: '+' '.join(metadata)
    print(f'--------------{i+1}-th combination------------------')
    run()'''


