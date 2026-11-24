import tensorflow as tf
import numpy as np
import obspy
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import random
import metrics as m

WINDOW_SIZE = 10000
WINDOW_CROSSOVER = 0.3
TEST_SIZE = 0.2 # percentage of data to test with (rest will train model)
random.seed(42)

# Labels: 0=noise, 1=avalanche, 2=earthquake
NOISE_LABEL = 0
AVA_LABEL = 1
EQ_LABEL = 2

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

    for i in range(0, len(data)-WINDOW_SIZE+1, int(WINDOW_SIZE*(1-WINDOW_CROSSOVER))):
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

random.shuffle(ava_files)

# Split up avalanche data into training and testing sets
ava_train_set = ava_files[0:int(0.8*len(ava_files))]
ava_test_set = ava_files[int(0.8*len(ava_files)):]

print(f"# of avalanche training files: {len(ava_train_set)}")
print(f"# of avalanche testing files: {len(ava_test_set)}")

eq_files = []
with open("eq_files.txt", "r") as text:
    for line in text:
        eq_files.append(line[:-1]) # Make list of all eq files

random.shuffle(eq_files)

# Split up earthquake data into training and testing sets
eq_train_set = eq_files[0:int(0.8*len(eq_files))]
eq_test_set = eq_files[int(0.8*len(eq_files)):]

print(f"# of earthquake training files: {len(eq_train_set)}")
print(f"# of earthquake testing files: {len(eq_test_set)}")

noise_files = []
with open("noise_files.txt", "r") as text:
    for line in text:
        noise_files.append(line[:-1]) # Make list of all noise files

random.shuffle(noise_files)

# Split up noise data into training and testing sets
noise_train_set = noise_files[0:int(0.8*len(noise_files))]
noise_test_set = noise_files[int(0.8*len(noise_files)):]

print(f"# of noise training files: {len(noise_train_set)}")
print(f"# of noise testing files: {len(noise_test_set)}")


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

# Acquire testing data from all three sets
ava_test_data, ava_test_samplerate = [], []
for file in ava_test_set:
    [data, samplerate] = load_mseed(file)
    ava_test_data.append(data)
    ava_test_samplerate.append(samplerate)

eq_test_data, eq_test_samplerate = [], []
for file in eq_test_set:
    [data, samplerate] = load_mseed(file)
    eq_test_data.append(data)
    eq_test_samplerate.append(samplerate)

noise_test_data, noise_test_samplerate = [], []
for file in noise_test_set:
    [data, samplerate] = load_mseed(file)
    noise_test_data.append(data)
    noise_test_samplerate.append(samplerate)


# Create windows of training data
train_windows, train_labels = [],[]
for i in range(len(ava_train_data)):
    windows = create_windows(ava_train_data[i])
    for window in windows: 
        train_windows.append(window)
        train_labels.append(AVA_LABEL)
for i in range(len(eq_train_data)):
    windows = create_windows(eq_train_data[i])
    for window in windows:
        train_windows.append(window)
        train_labels.append(EQ_LABEL)
for i in range(len(noise_train_data)):
    windows = create_windows(noise_train_data[i])
    for window in windows:
        train_windows.append(window)
        train_labels.append(NOISE_LABEL)

train_windows = np.array(train_windows, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int32)

# Create windows of testing data
test_windows, test_labels = [],[]
for i in range(len(ava_test_data)):
    windows = create_windows(ava_test_data[i])
    for window in windows:
        test_windows.append(window)
        test_labels.append(AVA_LABEL)
for i in range(len(eq_test_data)):
    windows = create_windows(eq_test_data[i])
    for window in windows:
        test_windows.append(window)
        test_labels.append(EQ_LABEL)
for i in range(len(noise_test_data)):
    windows = create_windows(noise_test_data[i])
    for window in windows:
        test_windows.append(window)
        test_labels.append(NOISE_LABEL)

test_windows = np.array(test_windows, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.int32)

# Implement Class Imbalance
w = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight = dict(enumerate(w))

print("Class Weights:", class_weight)

callbacks = [

    # Stop training if validation loss stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7, 
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save the best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_seismic_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when progress plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5, 
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Create model
model = tf.keras.Sequential([
    
    tf.keras.layers.Input(shape=(WINDOW_SIZE, 1)),

    tf.keras.layers.Conv1D(
        16, kernel_size=7,
        activation="relu"),

    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        32, kernel_size=5,
        activation="relu"),

    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(
        64, kernel_size=3,
        activation="relu"),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy",
             m.ClassF1(class_id=NOISE_LABEL, name="noise_F1"),
             m.ClassF1(class_id=AVA_LABEL, name="avalanche_F1"),
             m.ClassF1(class_id=EQ_LABEL, name="earthquake_F1")],
)

# Train model
history = model.fit(
    train_windows,
    train_labels,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    #callbacks=callbacks
)

# Evaluate model
loss, accuracy = model.evaluate(test_windows, test_labels)

print("Test accuracy:", accuracy)

# Evaluate model based on class not accuracy
pred = model.predict(test_windows).argmax(axis=1)

print(classification_report(test_labels, pred, target_names=['Noise', 'Avalanche', 'Earthquake']))
print(confusion_matrix(test_labels,pred))

"""
# Convert to tensorflow lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimixations = [
        tf.lite.Optimize.DEFAULT
    ]

tflite_model = converter.convert()

with open("seismic_classifier.tflite", "wb") as file:
    file.write(tflite_model)
"""
