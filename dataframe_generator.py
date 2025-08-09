import os
import json
import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

def get_image_time(img_name):
    """
    Get the time from the image file name.

    Args:
        img_name (str): Name of the image file.

    Returns:
        int: Time in microseconds.
    """

    img_data = img_name.split("_")
    hour = img_data[0]
    minute = img_data[1]
    second = img_data[2]
    micro = img_data[3].split(".")[0]

    time = int(hour) * 3600000000 + int(minute) * 60000000 + int(second) * 1000000 + int(micro)

    return time

def get_json_time(json_name):
    """
    Get the time from the JSON file name.

    Args:
        json_name (str): Name of the JSON file.

    Returns:
        int: Time in microseconds.
    """
    json_data = json_name.split("_")
    hour = json_data[0]
    minute = json_data[1]
    second = json_data[2]
    mili = json_data[3].split(".")[0]

    time = int(hour) * 3600000000 + int(minute) * 60000000 + (int(second)) * 1000000 + int(mili) * 1000

    return time

def process_accelerometer(accelerometer_data):
    """
    Process the accelerometer data and convert it into a numpy array.

    Args:
        accelerometer_data (list): List of accelerometer data.

    Returns:
        numpy.ndarray: Processed accelerometer data as a numpy array.
    """

    accelerometer_array = np.empty((0, 3))
    accelerometer_list = accelerometer_data[:100]

    for data in accelerometer_list:

        values = np.array([data["value0"], data["value1"], data["value2"]])
        accelerometer_array = np.append(accelerometer_array, [values], axis=0)

    return accelerometer_array

def process_gyroscope(gyroscope_data):
    """
    Process the gyroscope data and convert it into a numpy array.

    Args:
        gyroscope_data (list): List of gyroscope data.

    Returns:
        numpy.ndarray: Processed gyroscope data as a numpy array.
    """

    gyroscope_array = np.empty((0, 3))
    gyroscope_list = gyroscope_data[:100]

    for data in gyroscope_list:

        values = np.array([data["value0"], data["value1"], data["value2"]])
        gyroscope_array = np.append(gyroscope_array, [values], axis=0)

    return gyroscope_array

def process_rotation_vector(rotation_data):
    """
    Process the rotation data and convert it into a numpy array.

    Args:
        rotation_data (list): List of rotation data.

    Returns:
        numpy.ndarray: Processed rotation data as a numpy array.
    """

    rotation_array = np.empty((0, 4))
    rotation_list = rotation_data[:100]

    for data in rotation_list:

        values = np.array([data["value0"], data["value1"], data["value2"], data["value3"]])
        rotation_array = np.append(rotation_array, [values], axis=0)

    return rotation_array

def process_json_array(json_path):
    """
    Process the JSON data and extract the sensor data.

    Args:
        json_data (dict): JSON data.

    Returns:
        numpy.ndarray: Processed sensor data as a numpy array.
    """

    try:

        with open(json_path) as file:
            json_data = json.load(file)

    except FileNotFoundError:

        print(f"File {json_path} not found.")
        return None
    
    except json.JSONDecodeError:

        print(f"Error decoding JSON in file {json_path}.")
        return None

    data = json_data["data"]

    accelerometer_data = data["samsung_linear_acceleration_sensor"]
    gyroscope_data = data["lsm6dso_gyroscope"]
    rotation_data = data["samsung_rotation_vector"]

    accelerometer_array = process_accelerometer(accelerometer_data)
    gyroscope_array = process_gyroscope(gyroscope_data)
    rotation_array = process_rotation_vector(rotation_data)

    final_array = np.concatenate((accelerometer_array, gyroscope_array, rotation_array), axis=1)

    return final_array

def Process_Jsons_Path(path):
    """
    Process all the JSON files in the given path and extract the sensor data.

    Args:
        path (str): Path to the directory containing the JSON files.

    Returns:
        numpy.ndarray: Processed sensor data as a numpy array.
    """

    data = np.empty((0, 100, 10))

    for root, _, files in os.walk(path):

        for file in files:

            json_file = root + "/" + file

            if not json_file.endswith(".json"):
                continue

            data_array = process_json_array(json_file)
            data = np.append(data, [data_array], axis=0)

    return data

def Process_Json_DataFrame(path):
    """
    Generate a DataFrame from the JSON data.

    Args:
        path (str): Path to the JSON file.

    Returns:
        pandas.DataFrame: DataFrame containing the sensor data.
    """
    with open(path) as json_file:

        data_json = json.load(json_file)
        data = data_json["data"]

        accelerometer_data = data["samsung_linear_acceleration_sensor"]
        gyroscope_data = data["lsm6dso_gyroscope"]
        rotation_data = data["samsung_rotation_vector"]

        acc_dataframe = pd.DataFrame(accelerometer_data)
        gyro_dataframe = pd.DataFrame(gyroscope_data)
        rot_dataframe = pd.DataFrame(rotation_data)

        acc_dataframe.rename(columns={
            "timestamp": "Acc_Time",
            "value0": "Acc_X",
            "value1": "Acc_Y",
            "value2": "Acc_Z",
        }, inplace=True)

        gyro_dataframe.rename(columns={
            "timestamp": "Gyro_Time",
            "value0": "Gyro_X",
            "value1": "Gyro_Y",
            "value2": "Gyro_Z",
        }, inplace=True)

        rot_dataframe.rename(columns={
            "timestamp": "Rot_Time",
            "value0": "Rot_X",
            "value1": "Rot_Y",
            "value2": "Rot_Z",
            "value3": "Rot_W",
        }, inplace=True)

        rot_dataframe.drop(columns=["Value4"], inplace=True)

        dataframe = pd.concat([acc_dataframe, gyro_dataframe, rot_dataframe], axis=1)

        return dataframe 

def Generate_DataFrame(path):
    """
    Generate a DataFrame from the directory structure based on two types of data:
    'continuous' and 'instantaneous', with 'instantaneous' type having an additional
    directory level for gestures.

    Args:
        path (str): Path to the directory containing the data.

    Returns:
        pandas.DataFrame: DataFrame containing the data.
    """
    data = []
    data_types = ["continuous", "instantaneous"]

    for data_type in data_types:
        path_type = os.path.join(path, data_type)
        if not os.path.exists(path_type):
            continue

        list_subjects = sorted(os.listdir(path_type))

        for subject in list_subjects:
            
            path_subject = os.path.join(path_type, subject)
            if os.path.isfile(path_subject):
                continue

            list_actions = sorted(os.listdir(path_subject))

            for action in list_actions:
                path_action = os.path.join(path_subject, action)

                if data_type == "instantaneous":
                    list_gestures = sorted(os.listdir(path_action))
                    for gesture in list_gestures:
                        path_gesture = os.path.join(path_action, gesture)
                        data.append(process_entry(path_gesture, data_type, subject, action, gesture))
                else:
                    data.append(process_entry(path_action, data_type, subject, action))

    return pd.DataFrame(data)

def process_entry(base_path, data_type, subject, action, gesture=None):
    """
    Helper function to process each entry and construct the data dictionary.

    Args:
        base_path (str): The base path for the data entry.
        data_type (str): Type of data ('continuous' or 'instantaneous').
        subject (str): Subject identifier.
        action (str): Action identifier.
        gesture (str, optional): Gesture identifier. Defaults to None.

    Returns:
        dict: Data dictionary for a single row.
    """
    paths = {
        'camera_a': os.path.join(base_path, "camera_a"),
        'camera_b': os.path.join(base_path, "camera_b"),
        'sensors': os.path.join(base_path, "sensors")
    }

    if not all(os.path.exists(p) for p in paths.values()):
        return {}

    return {
        'type': data_type,
        'subject': subject,
        'action': action,
        'gesture': gesture if gesture else "None",
        'path_camera_A': paths['camera_a'],
        'path_camera_B': paths['camera_b'],
        'path_sensors': paths['sensors'],
        'camera_A': sorted(os.listdir(paths['camera_a'])),
        'camera_B': sorted(os.listdir(paths['camera_b'])),
        'sensors': sorted(os.listdir(paths['sensors']))
    }

def get_imgs_time(json, list_images):
    """
    Get the list of images and their times based on the JSON file.

    Args:
        json (str): Name of the JSON file.
        list_images (list): List of image names.

    Returns:
        tuple: Tuple containing the new list of images, list of image times, initial time, and final time.
    """

    time_json = get_json_time(json)

    init_json = time_json - 1000000

    new_list_images = [img for img in list_images if init_json < get_image_time(img) < time_json]
    list_time_images = [get_image_time(img) for img in new_list_images]

    return new_list_images, list_time_images, init_json, time_json

def Show_Json_Data(array, time):
    """
    Plot the accelerometer data from a numpy array and highlight a specific time,
    then display the plot as an image using OpenCV.

    Args:
        array (numpy.ndarray): Numpy array containing the accelerometer data.
        time (int): Time in microseconds to highlight on the plot.

    Raises:
        ValueError: If the array is empty or not a 2D array with at least three columns.
    """
    if array.size == 0 or array.shape[1] < 3:
        raise ValueError("Input array must be non-empty and have at least three columns.")

    # Create the plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    titles = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']

    # Plot each axis data with highlights
    for i in range(3):
        axs[i].plot(array[:, i])
        axs[i].axvline(x=time, color='r')
        axs[i].axhline(y=0, color='k')
        axs[i].set_title(titles[i], fontsize=20)  
        axs[i].set_ylim(np.min(array[:, i]) - 1, np.max(array[:, i]) + 1)

    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  


    # Convert buffer to OpenCV image
    buf.seek(0)
    img_array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img_json = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_json = cv2.resize(img_json, (640, 720))

    # Display the image using OpenCV
    cv2.imshow("Accelerometer Data", img_json)

def Visualize_Data(data):
    """
    Visualizes data by displaying images and associated JSON accelerometer data.

    Args:
        data (dict): A dictionary containing the following keys:
            - "path_sensors" (str): The path to the sensor JSON files.
            - "sensors" (list): A list of sensor JSON file names.
            - "path_camera_A" (str): The path to the camera A images.
            - "camera_A" (list): A list of camera A image file names.

    Returns:
        None
    """

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1280, 720)

    path_json = data["path_sensors"]
    list_jsons = data["sensors"]
    path_images = data["path_camera_A"]
    list_images = data["camera_A"]

    for json in list_jsons:
        json_file = os.path.join(path_json, json)

        try:
            array = process_json_array(json_file)
        except FileNotFoundError:
            print(f"File not found: {json_file}")
            continue

        new_list_images, list_time_images, init_json, time_json = get_imgs_time(json, list_images)

        for img_name, img_time in zip(new_list_images, list_time_images):
            image_path = os.path.join(path_images, img_name)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                image = cv2.resize(image, (1280, 720))
                cv2.imshow('Image', image)

                time = int((img_time - init_json) / (time_json - init_json) * 100)
                Show_Json_Data(array, time)

                if cv2.waitKey(1) == 27:  # Exit on ESC key
                    break
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")
                continue

    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Generate the dataframe
    data_base_path = "G:/CADDI_dataset" # Path to the data
    data_frame_path = "./dataframe.h5" # Path to save the DataFrame

    # Generate the DataFrame
    print("Generating the DataFrame", "\n")
    dataframe = Generate_DataFrame(data_base_path)

    # Saving to HDF5
    dataframe.to_hdf(data_frame_path, key="data", mode="w")
    print("DataFrame generated and saved to", data_frame_path, "\n")

    # Read the dataframe
    dataframe = pd.read_hdf(data_frame_path)

    #First instance data
    data = dataframe.iloc[0]

    print("First instance data", "\n")
    print(data)
    print("\n", "*"*50, "\n")

    # Process the JSON data
    json_path = data["path_sensors"] + "/" + data["sensors"][0]

    # Process the JSON as a DataFram
    json_dataframe = Process_Json_DataFrame(json_path)

    print("Processed JSON as a DataFrame", "\n")
    print(json_dataframe.head())
    print("\n", "*"*50, "\n")

    # Process the JSON as a numpy array
    data_array = process_json_array(json_path)

    print("Processed JSON as a numpy array", "\n")
    print(data_array.shape, "\n")
    print(data_array[:5])
    print("\n", "*"*50, "\n")

    # Process all the JSON files in the given path as a numpy array
    path = data["path_sensors"]
    data_array = Process_Jsons_Path(path)

    print("Processed all JSON files in the given path", "\n")
    print(data_array.shape, "\n")
    print(data_array[:2])

    #Visualize the data of subject 4, action drink, gesture 5
    subject = "subject_4"
    action = "drink"
    gesture = "gesture_5"

    sub_dataframe = dataframe[(dataframe["subject"] == subject) & (dataframe["action"] == action) & (dataframe["gesture"] == gesture)]

    if len(sub_dataframe) > 0:

        data = sub_dataframe.iloc[0]
        Visualize_Data(data)

