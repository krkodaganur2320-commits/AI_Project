from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(
    model_path="models/model_unquant.tflite"
)

interpreter.allocate_tensors()

print("MODEL LOADED SUCCESSFULLY")

print("INPUT DETAILS:")
print(interpreter.get_input_details())

print("OUTPUT DETAILS:")
print(interpreter.get_output_details())
