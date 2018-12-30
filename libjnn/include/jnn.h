#ifndef JNN_JNN_H
#define JNN_JNN_H

#include <stdint.h>

#ifdef __cplusplus
  #define C_API extern "C"
#else
  #define C_API
#endif

typedef uint8_t ErrorCode;
typedef void* JnnPtr;

/*!
 * \brief Type of activation functions supported
 */
enum JnnActFn {
  JNN_IDENTITY,
  JNN_SIGMOID,
  JNN_SOFTMAX
};

/*!
 *
 * \brief Create a new Jnn handler
 * 
 * \return An opaque pointer to Jnn. NULL if operation failed.
 */
C_API JnnPtr jnn_new();

/*!
 * \brief Release all resources used by the Jnn pointed by the handler
 */
C_API ErrorCode jnn_delete(JnnPtr jnn);

/*!
 *
 */
C_API ErrorCode jnn_set_input(JnnPtr jnn, uint64_t units);

/*!
 *
 * Dense, fully connected, layer uses RELU as its activation function
 */
C_API ErrorCode jnn_add_dense_layer(JnnPtr jnn, uint64_t units);

/*!
 *
 */
C_API ErrorCode jnn_add_output(JnnPtr jnn,
                               uint64_t units,
                               enum JnnActFn fun);

/*!
 *
 */
C_API ErrorCode jnn_forward(JnnPtr jnn,
                            float* data,
                            uint64_t n,
                            uint64_t feature_dim,
                            float* output);



#endif // JNN_JNN_H
