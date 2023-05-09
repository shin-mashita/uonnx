#include "uonnx_utils.h"

// Extras
char * TensorType2String(TensorType t) 
{
    switch (t)
    {
        case TENSOR_TYPE_UNDEFINED	:
            return "UNDEFINED";
            break;
        case TENSOR_TYPE_BOOL		:
            return "BOOL";
            break;
        case TENSOR_TYPE_INT8		:
            return "INT8";
            break;
        case TENSOR_TYPE_INT16		:
            return "INT16";
            break;
        case TENSOR_TYPE_INT32		:
            return "INT32";
            break;
        case TENSOR_TYPE_INT64		:
            return "INT64";
            break;
        case TENSOR_TYPE_UINT8		:
            return "UINT8";
            break;
        case TENSOR_TYPE_UINT16		:
            return "UINT16";
            break;
        case TENSOR_TYPE_UINT32		:
            return "UINT32";
            break;
        case TENSOR_TYPE_UINT64		:
            return "UINT64";
            break;
        case TENSOR_TYPE_BFLOAT16	:
            return "BFLOAT16";
            break;
        case TENSOR_TYPE_FLOAT16	:
            return "FLOAT16";
            break;	
        case TENSOR_TYPE_FLOAT32	:
            return "FLOAT32";
            break;	
        case TENSOR_TYPE_FLOAT64	:
            return "FLOAT64";
            break;
        case TENSOR_TYPE_COMPLEX64	:
            return "COMPLEX64";
            break;
        case TENSOR_TYPE_COMPLEX128	:
            return "COMPLEX128";
            break;
        case TENSOR_TYPE_STRING		:
            return "STRING";
            break;
        default:
            return NULL;
            break;
    }
}

