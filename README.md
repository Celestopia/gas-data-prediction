# Time Series Analysis Library of Gas Data Prediction

(A library for personal study, not developed yet)

(Can also be applied to other types of time series data)

## Guidance

The main function is in `main.py`. The components of main function is defined in `components.py`.

Run `run.py` to conduct experiments. Loops in `run.py` can be customized to run experiments with different hyperparameters.

`extract_json.py` is used to extract the experiment data stored in .json files under a directory, and summarize them into an Excel file.

Currently, the dataset is stored in a `.pkl` file. You can also modify `data_provider\data_reading.py` to modify the `load_data` function to fit your own dataset.

`flame_mechanism_20250228_1.pkl` stores the 3-d grid of flame speed and flame temperature data, which can be used for interpolation. It is generated from `feature_reconstruction\generate_flame_temp_speed_20250228_1.ipynb`. No need to run it again, since it requires about 11 hours.

Use `reconstruct.py` to interpolate the flame temperature and flame speed data using the 3-d grid, and save the dataset as a `.pkl` file.

**For detailed introduction, see the annotations of each file.**




## Overall logic

- `run.py`: Run customized loops of experiments.
    - `main.py`: The whole process of one experiment, including data reading, data processing, model training, model evaluation, and result saving.
        - `components.py`: Component functions of main function, including:
            - `get_dataset()`: Get everything needed of the dataset, e.g. data, variable names, variable units, etc.
                - `data_provider\data_reading.py load_data()`: Called in `get_dataset()`. Return the dataset, variable names, variable units. Can be customized.
            - `model_building_and_training()`: Build, train, and save a model.
            - `model_evaluation()`: Evaluate the performance of a model.
            - `save_plots()`: Save plots of model prediction performance.
            - `save_result()`: Save model experiment information to result.json
            - `save_objects()`: Save the model and experiment objects as a .pkl file, in order that they can be loaded later for further analysis or visualization. An example of further analysis is `others\make_figure.ipynb`.









