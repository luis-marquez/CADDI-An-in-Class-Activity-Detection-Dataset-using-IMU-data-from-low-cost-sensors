# CADDI-An-in-Class-Activity-Detection-Dataset-using-IMU-data-from-low-cost-sensors
This repository contains a Python script (dataframe_generator.py) that processes sensor data from JSON files and generates a structured pandas DataFrame. The script is designed to work with accelerometer, gyroscope, and rotation vector data, as well as visual data from two cameras. It provides the code to load, process, and visualize this data.

# Dataframe Generator

This repository contains a Python script (`dataframe_generator.py`) that processes sensor data from JSON files and generates a structured **pandas DataFrame**.  
The script is designed to handle **accelerometer**, **gyroscope**, and **rotation vector** data, as well as visual data from **two cameras**.  
It provides utilities for loading, processing, and visualizing this data.

---

## **Features**

- **Sensor Data Processing**: Handles accelerometer, gyroscope, and rotation vector data, converting them into a usable format.
- **Data Visualization**: Provides functions to visualize sensor data alongside corresponding camera images.
- **DataFrame Generation**: Combines sensor data and stores it in a pandas DataFrame for further analysis.
- **Support for Continuous and Instantaneous Data**: Processes two distinct data structures with different handling paths.

---

## **Dependencies**

The script requires the following Python libraries:

- `os`
- `json`
- `numpy`
- `pandas`
- `cv2` (OpenCV)
- `matplotlib`

Install the required libraries with:

```bash
pip install numpy pandas opencv-python matplotlib
```


## **Usage**
1. Set the base path for the dataset and run the script:

    ```bash
    data_base_path = "path_to_dataset"
    data_frame_path = "./dataframe.h5"
    dataframe = Generate_DataFrame(data_base_path)
    dataframe.to_hdf(data_frame_path, key="data", mode="w")
    ```
2. To visualize data, ensure the images and sensor data are correctly organized, and call the ```Visualize_Data()``` function with the appropriate subject, action, and gesture.

3. The script can also generate and process JSON files into a pandas DataFrame:
```
json_dataframe = Process_Json_DataFrame(json_path)
```

## **File Structure**
dataframe_generator.py: Main and unique file that handles data processing and visualization.


## **Data Access**
``https://www.scidb.cn/en/detail?dataSetId=46e450feed9a4259a070a16db39eed93```


## **Citation**
Marquez-Carpintero, L., Suescun-Ferrandiz, S., Pina-Navarro, M., Cazorla, M., & Gomez-Donoso, F. (2025).
_CADDI: An in-Class Activity Detection Dataset using IMU data from low-cost sensors._
arXiv preprint arXiv:2503.02853.

