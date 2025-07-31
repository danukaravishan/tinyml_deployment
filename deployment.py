import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os


def convert_onnx_to_tflite(onnx_model_path="models/model.onnx", 
                          tflite_output_path="models/model.tflite"):
    
    saved_model_dir = "models/saved_model"
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS      # standard TFLite ops
        #tf.lite.OpsSet.SELECT_TF_OPS         # allow unsupported TF ops like AddV2
    ]

    # Optional: improve model compatibility
    converter.experimental_enable_resource_variables = True

    # Optional: apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert model
    tflite_model = converter.convert()

    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    return 0



def test_tflite_model(tflite_model_path="models/deployment/model.tflite"):
    try:
        import tensorflow as tf
        
        print(f"Testing TFLite model: {tflite_model_path}")
        
        # Load TFLite model
        # Load TFLite model with optional Flex delegate (for custom TF ops)
        delegates = []
        try:
            from tflite_runtime.interpreter import load_delegate
            flex_delegate = load_delegate('libtensorflowlite_flex_delegate.so')
            delegates.append(flex_delegate)
            print('✨ Flex delegate loaded')
        except Exception:
            print('⚠ Flex delegate not available, running without it')
            interpreter = Interpreter(
            model_path=tflite_model_path,
            experimental_delegates=delegates if delegates else None
        )
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output type: {output_details[0]['dtype']}")
        
        # Create test input
        input_shape = input_details[0]['shape']
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        # Run inference
        import time
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        print(f"✓ TFLite inference successful!")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Output value: {output_data}")
        print(f"Output shape: {output_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ TFLite test failed: {e}")
        return False



########### Test the model accuracy ##########
def test_tflite_accuracy(tflite_model_path="models/deployment/model.tflite"):  

    import h5py
    # Try lightweight tflite-runtime first, otherwise fall back to full TensorFlow
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        load_delegate = None

    def extractDataFromHDF5Group(group):
        data_list = []
        for event_id in group.keys():
            dataset = group[event_id]
            data = np.array(dataset)  # Convert dataset to NumPy array
            data_list.append(data)    # Append to list
        return data_list

    def getWaveData(hdf5_file):
        positive_group_p = hdf5_file['positive_samples_p']
        negative_group   = hdf5_file['negative_sample_group']

        p_data = extractDataFromHDF5Group(positive_group_p)
        noise_data= extractDataFromHDF5Group(negative_group)

        hdf5_file.close()

        return p_data, noise_data

    def normalize(signal):
        """Normalize the signal between -1 and 1."""
        min_val = np.min(signal)
        max_val = np.max(signal)
    # Initialize TFLite interpreter, loading Flex delegate if available
    delegates = []
    if load_delegate:
        try:
            flex = load_delegate('libtensorflowlite_flex_delegate.so')
            delegates.append(flex)
            print("✨ Flex delegate loaded for custom ops")
        except Exception as e:
            print(f"⚠ Flex delegate load failed: {e}")
    interpreter = Interpreter(
        model_path=tflite_model_path,
        experimental_delegates=delegates or None
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
        
    def normalize_data(data):
        if data.ndim == 3:  # Case for a set of data
            processed_data = []
            for sig in data:
                normalised = np.array([normalize(sig[i, :]) for i in range(sig.shape[0])])  # Normalize each component
                processed_data.append(normalised)
            return np.array(processed_data)
        elif data.ndim == 2:  # Case for a single input
            return np.array([normalize(data[i, :]) for i in range(data.shape[0])])  # Normalize each component
        else:
            raise ValueError("Input data must have 2 or 3 dimensions.")
    
    
    # Load TFLite model with optional Flex delegate for unsupported TF ops
    from tflite_runtime.interpreter import load_delegate
    delegates = []
    try:
        flex_delegate = load_delegate('libtensorflowlite_flex_delegate.so')
        delegates.append(flex_delegate)
        print("✨ Flex delegate loaded")
    except Exception as e:
        print("⚠ Flex delegate not found, running without it:", e)
    interpreter = Interpreter(
        model_path=tflite_model_path,
        experimental_delegates=delegates or None
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare data
    print("Loading test data from HDF5...")
    hdf5_file = h5py.File("data/test_data.h5", 'r')
    p_data, noise_data = getWaveData(hdf5_file)
    p_data = np.array(p_data)
    noise_data = np.array(noise_data)
    p_data = normalize_data(p_data)
    noise_data = normalize_data(noise_data)
    
    # Balance classes
    n_p = len(p_data)
    if len(noise_data) > n_p:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(noise_data), n_p, replace=False)
        noise_data = noise_data[idx]
    
    # Construct test set and labels
    test_data = np.concatenate((p_data, noise_data))
    true_labels = np.array([1]*len(p_data) + [0]*len(noise_data))
    
    # Run inference
    print(f"Running TFLite inference on {len(test_data)} samples...")
    predictions = []
    for sample in test_data:
        inp = sample.reshape(1, 3, 100).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])[0]
        # Scalar output
        if out.ndim > 0:
            predictions.append(out.flatten()[0])
        else:
            predictions.append(float(out))
    predictions = np.array(predictions)
    pred_classes = (predictions > 0.57).astype(int)
    
    # Compute metrics
    TP = int(((pred_classes == 1) & (true_labels == 1)).sum())
    TN = int(((pred_classes == 0) & (true_labels == 0)).sum())
    FP = int(((pred_classes == 1) & (true_labels == 0)).sum())
    FN = int(((pred_classes == 0) & (true_labels == 1)).sum())
    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
    precision = 100 * TP / (TP + FP) if (TP + FP)>0 else 0.0
    recall = 100 * TP / (TP + FN) if (TP + FN)>0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall)>0 else 0.0
    
    # Print
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    
    convert_onnx_to_tflite()
    #test_tflite_model()
    #test_tflite_accuracy()
