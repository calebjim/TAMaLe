import tensorflow as tf
import numpy as np
import obspy

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

ava_files = []
with open("ava_files.txt", "r") as text:
    for line in text:
        ava_files.append(line) # Make list of all ava files

# Split up avalanche data into training and testing sets
ava_train_set = ava_files[0:int(0.8*len(ava_files))]
ava_test_set = ava_files[int(0.8*len(ava_files)):]

print(len(ava_train_set))
print(len(ava_test_set))

eq_files = []
with open("eq_files.txt", "r") as text:
    for line in text:
        eq_files.append(line) # Make list of all eq files

# Split up earthquake data into training and testing sets
eq_train_set = eq_files[0:int(0.8*len(eq_files))]
eq_test_set = eq_files[int(0.8*len(eq_files)):]

print(len(eq_train_set))
print(len(eq_test_set))

noise_files = []
with open("noise_files.txt", "r") as text:
    for line in text:
        noise_files.append(line) # Make list of all noise files

# Split up noise data into training and testing sets
noise_train_set = noise_files[0:int(0.8*len(noise_files))]
noise_test_set = noise_files[int(0.8*len(noise_files)):]

print(len(noise_train_set))
print(len(noise_test_set))
