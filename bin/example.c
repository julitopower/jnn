#include <stdio.h>

#include <jnn.h>

int main(int argc, char** argv) {
  printf("Julio's Neural Network (jNN) example\n");

  JnnPtr jnn = jnn_new();
  jnn_add_output(jnn, 3, JNN_IDENTITY);
  jnn_add_output(jnn, 10, JNN_IDENTITY);
  jnn_add_output(jnn, 100, JNN_IDENTITY);    

  jnn_delete(jnn);
}
