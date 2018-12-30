#include <jnn.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

////////////////////////////////////////////////////////////////////////////////
// Activation functions
////////////////////////////////////////////////////////////////////////////////
typedef ErrorCode (*ActivationFunction)(float*, uint64_t, float*);

/*
 * Identity activation function. It basically copies the activations
 * into the output. This sort of activation function is useful for
 * regression.
 */
static ErrorCode jnn_identity(float* in, uint64_t n, float* out) {
  memcpy(out, in, n * sizeof(float));
  return 0;
}

// Activation functions

////////////////////////////////////////////////////////////////////////////////
// Layer datastructure and related functions
////////////////////////////////////////////////////////////////////////////////
typedef struct Layer Layer;
struct Layer {
  uint64_t           inputs;            // number of inputs
  uint64_t           units;             // number of units
  float*             weights;           // weights matrix
  float*             outputs;           // outputs vector
  float*             activations;       // activations vector
  ActivationFunction activation_function;
  
  Layer*             next;              // pointer to next layer
  Layer*             prev;              // pointer to previous layer
};

/*
 * Auxiliary function to create a new layer.
 */
static Layer* jnn_new_layer(uint64_t inputs,
                            uint64_t units,
                            enum JnnActFn activation_fn) {
  Layer* layer = (Layer*)malloc(1 * sizeof(Layer));
  if (layer == NULL) {
    return layer;
  }
                         
  layer->inputs = inputs;
  layer->units = units;

  // TODO: Add memory allocation checks
  layer->weights = (float*) calloc(inputs * units, sizeof(float));
  // TODO:  Initialize the weights with N(0, 1)
  srand(time(NULL));
  for (uint64_t i = 0 ; i < inputs * units ; ++i) {
    layer->weights[i] = ((float)rand()) / RAND_MAX;
  }
  
  layer->outputs = (float*) calloc(units, sizeof(float));
  layer->activations = (float*) calloc(units, sizeof(float));

  // TODO: Set pointer to the actual activation function
  layer->activation_function = jnn_identity;
  layer->next = NULL;
  layer->prev = NULL;
  return layer;
}

static ErrorCode jnn_delete_layer(Layer* layer) {
  if (layer != NULL) {
    free(layer->weights);
    free(layer->outputs);
    free(layer->activations);
    free(layer);
  }
  return 0;
}

// Layer related code

typedef struct Jnn {
  uint64_t layers;
  uint64_t outputs;
  Layer* layer_ptr;
  
} Jnn;

/*
 * Auxiliary function to add a new layer to a JNN
 */
static ErrorCode jnn_add_layer(JnnPtr jnn,
                               uint64_t units,
                               enum JnnActFn fun) {
  Jnn* handler = jnn;
  // Create the new layer and link it to the Jnn
  uint64_t inputs  = (handler->layer_ptr == NULL) ? units : handler->outputs;
  Layer* output = jnn_new_layer(inputs, units, fun);
  if (handler->layer_ptr == NULL) {
    handler->layer_ptr = output;
  } else {
    Layer* curr = handler->layer_ptr;
    while (curr->next != NULL) {
      curr = curr->next;
    }
    curr->next = output;
    output->prev = curr;
  }

  // Update Jnn metadata
  handler->layers++;
  handler->outputs = units;

  return 0;
}

C_API JnnPtr jnn_new() {
  Jnn* handler = (Jnn*)malloc(1 * sizeof(Jnn));
  if (!handler) { 
    return handler;
  }

  handler->layers = 0;
  handler->outputs = 0;
  handler->layer_ptr = NULL;
  return handler;
}

C_API ErrorCode jnn_set_input(JnnPtr jnn, uint64_t units) {
  Jnn* handler = jnn;
  handler->outputs = units;
  return 0;
}

C_API ErrorCode jnn_add_output(JnnPtr jnn,
                         uint64_t units,
                         enum JnnActFn fun) {
  return jnn_add_layer(jnn, units, fun);
}

C_API ErrorCode jnn_delete(JnnPtr jnn) {
  if (jnn == NULL) {
    return 0;
  }

  Jnn* handler = jnn;
  Layer* curr = handler->layer_ptr;
  while (curr != NULL && curr->next != NULL) {
    curr = curr->next;
  }

  if (curr != NULL) {
    while (curr != NULL) {
      Layer* prev = curr->prev;
      // TODO: Need cleanup function for layers, this one leaks weights, ...
      jnn_delete_layer(curr);
      curr = prev;
    }
  }

  free(handler);
  return 0;
}

C_API ErrorCode jnn_add_dense_layer(JnnPtr jnn, uint64_t units) {
  return jnn_add_layer(jnn, units, JNN_SIGMOID);
}

C_API ErrorCode jnn_forward(JnnPtr jnn,
                            float* data,
                            uint64_t n,
                            uint64_t feature_dim,
                            float* output) {

  /*
    for_each(layer) {
        activations = matrix_mul(input, weights);
        layer_output = activation_function(activations);
        input = layer_output
    }

    output = input;

    Lost of good code in https://github.com/tiny-dnn/tiny-dnn/blob/1c5259477b8b4eab376cc19fd1d55ae965ef5e5a/tiny_dnn/core/kernels/fully_connected_op_internal.h#L15 too
    
    Use https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py for guidance
   */
  
  return 0;
}
