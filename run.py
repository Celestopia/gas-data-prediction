from main import main, param_dict
from types import SimpleNamespace
args = SimpleNamespace(**param_dict) # Convert the dictionary to a namespace object for easier access to the parameters.


# Customize the hyperparameters
args.data_path = r'E:\PythonProjects\gas-data-prediction\feature_reconstruction\data1344.pkl'
args.seed = 1234
args.result_directory = "./results20250302"
args.model_name = "SVR"
args.input_var_names = [
        "Environment Temperature",
        "Pumping Flow",
        "Inside Temperature",
        "Inside Humidity",
        "Inside Pressure",
        "Outside Temperature",
        "Outside Humidity",
        "Outside Pressure",
        "FGR",
        "Fan Frequency",
        "Load",
        "Gas Bias Valve",
        "Fan Bias Valve",
        #"O2",
        #"Flame Temperature",
        #"Flame Speed",
    ]
args.output_var_names = [
        #"O2",
        "CO2",
        "NOx",
        "Smoke Temperature",]

args.kernel = 'rbf'

for i in [10,30,60,90,120]: # Loop over a hyperparameter
    args.input_len=i # Change the input length
    main(args)
