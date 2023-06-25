#ifndef __UONNX_LOADER_H__
#define __UONNX_LOADER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <uonnx.h>

/**
 * @brief Load ModelProto from buffer from header
 * 
 * @param buf Buffer data
 * @param len Size of buffer in bytes
 * @return ModelProto* 
 */
ModelProto * load_model_buf(const void * buf, size_t len);

/**
 * @brief Load model from model.onnx
 * 
 * @param filename Filename of model.onnx
 * @return ModelProto* 
 */
ModelProto * load_model(const char * filename);

/**
 * @brief Free ModelProto model
 * 
 * @param model 
 */
void free_model(ModelProto * model);


#ifdef __cplusplus
}
#endif


#endif