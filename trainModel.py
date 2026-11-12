import tensorflow as tf
import numpy as np
import obspy

WINDOW_SIZE = 100

def load_mseed(file_path):
    """
    Load MiniSEED file and return the waveform data as a NumPy array.
    """
    try:
        st = obspy.read(file_path) # Read mseed file
    except Exception as e:
        raise RuntimeError(f"Error reading .mseed file")

    st.merge(method=1, fill_value=0)
    trace = st[0]
    data = trace.data.astype(np.float32)
    return data, trace.stats.sampling_rate

def create_windows(data):
    """
    Splits data into fixed-length windows
    """
    windows = []

    for i in range(0, len(data)-WINDOW_SIZE+1, WINDOW_SIZE):
        window = data[i:i+WINDOW_SIZE]
        
        # Normalize windows
        mean = np.mean(window)
        std = np.std(window)

        if std > 0:
            window = (window-mean)/std

        windows.append(window)

    return windows

ava_files = []
with open("ava_files.txt", "r") as text:
    for line in text:
        ava_files.append(line[:-1]) # Make list of all ava files
"""
# Split up avalanche data into training and testing sets
ava_train_set = ava_files[0:int(0.8*len(ava_files))]
ava_test_set = ava_files[int(0.8*len(ava_files)):]

print(f"# of avalanche training files: {len(ava_train_set)}")
print(f"# of avalanche testing files: {len(ava_test_set)}")
"""
eq_files = []
with open("eq_files.txt", "r") as text:
    for line in text:
        eq_files.append(line[:-1]) # Make list of all eq files
"""
# Split up earthquake data into training and testing sets
eq_train_set = eq_files[0:int(0.8*len(eq_files))]
eq_test_set = eq_files[int(0.8*len(eq_files)):]

print(f"# of earthquake training files: {len(eq_train_set)}")
print(f"# of earthquake testing files: {len(eq_test_set)}")
"""
noise_files = []
with open("noise_files.txt", "r") as text:
    for line in text:
        noise_files.append(line[:-1]) # Make list of all noise files
"""
# Split up noise data into training and testing sets
noise_train_set = noise_files[0:int(0.8*len(noise_files))]
noise_test_set = noise_files[int(0.8*len(noise_files)):]

print(f"# of noise training files: {len(noise_train_set)}")
print(f"# of noise testing files: {len(noise_test_set)}")
"""
"""
# Acquire training data from all three sets
ava_train_data, ava_train_samplerate = [], []
for file in ava_train_set:
    [data, samplerate] = load_mseed(file)
    ava_train_data.append(data)
    ava_train_samplerate.append(samplerate)

eq_train_data, eq_train_samplerate = [], []
for file in eq_train_set:
    [data, samplerate] = load_mseed(file)
    eq_train_data.append(data)
    eq_train_samplerate.append(samplerate)

noise_train_data, noise_train_samplerate = [], []
for file in noise_train_set:
    [data, samplerate] = load_mseed(file)
    noise_train_data.append(data)
    noise_train_samplerate.append(samplerate)
"""
# Turn all data into np arrays
ava_data, ava_samplerate = [], []
for file in ava_files:
    [data, samplerate] = load_mseed(file)
    ava_data.append(data)
    ava_samplerate.append(samplerate)

eq_data, eq_samplerate = [], []
for file in eq_files:
    [data, samplerate] = load_mseed(file)
    eq_data.append(data)
    eq_samplerate.append(samplerate)

noise_data, noise_samplerate = [], []
for file in noise_files:
    [data, samplerate] = load_mseed(file)
    noise_data.append(data)
    noise_samplerate.append(samplerate)

# Create windows of data
ava_windows,eq_windows,noise_windows = [],[],[]
for i in range(len(ava_data)):
    windows = create_windows(ava_data[i])
    for window in windows:
        ava_windows.append(window)
for i in range(len(eq_data)):
    windows = create_windows(eq_data[i])
    for window in windows:
        eq_windows.append(window)
for i in range(len(noise_data)):
    windows = create_windows(noise_data[i])
    for window in windows:
        noise_windows.append(window)

