#include <Arduino.h>
#include "sin_predictor.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define INPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 7
#define OUTPUT_BUFFER_SIZE 64

char in_str_buff[INPUT_BUFFER_SIZE];
int input_array[INT_ARRAY_SIZE];
int in_buff_idx = 0;

namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 2 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers = 0;
  char *token = strtok(in_str, ",");
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) break;
  }
  return num_integers;
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  delay(2000);
  model = tflite::GetModel(sin_predictor_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  if (Serial.available()) {
    char received_char = Serial.read();
    if (received_char == 13) {
      int array_length = string_to_array(in_str_buff, input_array);
      if (array_length == INT_ARRAY_SIZE) {
        for (int i = 0; i < INT_ARRAY_SIZE; i++) {
          input->data.int8[i] = (int8_t)input_array[i];
        }
        unsigned long start_time = micros();
        interpreter->Invoke();
        unsigned long end_time = micros();

        Serial.print("Prediction: ");
        Serial.println((int)output->data.int8[0]);
        Serial.print("Inference Time (µs): ");
        Serial.println(end_time - start_time);
      } else {
        Serial.println("Please enter exactly 7 integers.");
      }
      memset(in_str_buff, 0, INPUT_BUFFER_SIZE);
      in_buff_idx = 0;
    } else {
      in_str_buff[in_buff_idx++] = received_char;
    }
  }
}
