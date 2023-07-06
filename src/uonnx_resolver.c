/**
 * Reference from https://github.com/xboot/libonnx
 * 
 */

#include "uonnx_resolver.h"

void resolver_solve_operator(Resolver * r, Node * n)
{
    void (*rop)(Node *);

    if(r && n)
    {
        switch(shash(n->proto->op_type)) // Changed from 
        {
            case 0x0b87d47b: /* "Abs" */
                rop = r->op_Abs;
                break;
            case 0x7c82680b: /* "Acos" */
                rop = r->op_Acos;
                break;
            case 0x0ccf69d3: /* "Acosh" */
                rop = r->op_Acosh;
                break;
            case 0x0b87d4ae: /* "Add" */
                rop = r->op_Add;
                break;
            case 0x0b87d5f8: /* "And" */
                rop = r->op_And;
                break;
            case 0xa7c70ea5: /* "ArgMax" */
                rop = r->op_ArgMax;
                break;
            case 0xa7c70fa3: /* "ArgMin" */
                rop = r->op_ArgMin;
                break;
            case 0x7c82ab50: /* "Asin" */
                rop = r->op_Asin;
                break;
            case 0x0cd815b8: /* "Asinh" */
                rop = r->op_Asinh;
                break;
            case 0x7c82ae89: /* "Atan" */
                rop = r->op_Atan;
                break;
            case 0x0cd88011: /* "Atanh" */
                rop = r->op_Atanh;
                break;
            case 0xf1a1e23a: /* "AveragePool" */
                rop = r->op_AveragePool;
                break;
            case 0x2d3b46ee: /* "BatchNormalization" */
                rop = r->op_BatchNormalization;
                break;
            case 0x0bfe45a2: /* "BitShift" */
                rop = r->op_BitShift;
                break;
            case 0x7c8378d0: /* "Cast" */
                rop = r->op_Cast;
                break;
            case 0x7c838882: /* "Ceil" */
                rop = r->op_Ceil;
                break;
            case 0x7c83a64d: /* "Clip" */
                rop = r->op_Clip;
                break;
            case 0xb7db9db1: /* "Compress" */
                rop = r->op_Compress;
                break;
            case 0xac3f4a9d: /* "Concat" */
                rop = r->op_Concat;
                break;
            case 0x5053caca: /* "ConcatFromSequence" */
                rop = r->op_ConcatFromSequence;
                break;
            case 0xba6816ef: /* "Constant" */
                rop = r->op_Constant;
                break;
            case 0xe468a875: /* "ConstantOfShape" */
                rop = r->op_ConstantOfShape;
                break;
            case 0x7c83b3bb: /* "Conv" */
                rop = r->op_Conv;
                break;
            case 0x8371dbe9: /* "ConvInteger" */
                rop = r->op_ConvInteger;
                break;
            case 0x3903c4ba: /* "ConvTranspose" */
                rop = r->op_ConvTranspose;
                break;
            case 0x0b87deaa: /* "Cos" */
                rop = r->op_Cos;
                break;
            case 0x7c83b452: /* "Cosh" */
                rop = r->op_Cosh;
                break;
            case 0xacab0fbf: /* "CumSum" */
                rop = r->op_CumSum;
                break;
            case 0xc9c1d669: /* "DepthToSpace" */
                rop = r->op_DepthToSpace;
                break;
            case 0xf9cc985a: /* "DequantizeLinear" */
                rop = r->op_DequantizeLinear;
                break;
            case 0x0b87e1a2: /* "Det" */
                rop = r->op_Det;
                break;
            case 0x0b87e228: /* "Div" */
                rop = r->op_Div;
                break;
            case 0x883bca72: /* "Dropout" */
                rop = r->op_Dropout;
                break;
            case 0xb07d4f76: /* "Einsum" */
                rop = r->op_Einsum;
                break;
            case 0x0b87e6cb: /* "Elu" */
                rop = r->op_Elu;
                break;
            case 0x0d1f905d: /* "Equal" */
                rop = r->op_Equal;
                break;
            case 0x0b87e782: /* "Erf" */
                rop = r->op_Erf;
                break;
            case 0x0b87e852: /* "Exp" */
                rop = r->op_Exp;
                break;
            case 0xb18d8a45: /* "Expand" */
                rop = r->op_Expand;
                break;
            case 0xe4c1560d: /* "EyeLike" */
                rop = r->op_EyeLike;
                break;
            case 0x13363dd3: /* "Flatten" */
                rop = r->op_Flatten;
                break;
            case 0x0d2ed347: /* "Floor" */
                rop = r->op_Floor;
                break;
            case 0x0b87ebd3: /* "GRU" */
                rop = r->op_GRU;
                break;
            case 0xb499f620: /* "Gather" */
                rop = r->op_Gather;
                break;
            case 0x7c94d43d: /* "GatherElements" */
                rop = r->op_GatherElements;
                break;
            case 0x42f00872: /* "GatherND" */
                rop = r->op_GatherND;
                break;
            case 0x7c85ba8b: /* "Gemm" */
                rop = r->op_Gemm;
                break;
            case 0x9289c84b: /* "GlobalAveragePool" */
                rop = r->op_GlobalAveragePool;
                break;
            case 0x3f5a29ac: /* "GlobalLpPool" */
                rop = r->op_GlobalLpPool;
                break;
            case 0x575f0fb6: /* "GlobalMaxPool" */
                rop = r->op_GlobalMaxPool;
                break;
            case 0x6e6d652f: /* "Greater" */
                rop = r->op_Greater;
                break;
            case 0x10341df0: /* "HardSigmoid" */
                rop = r->op_HardSigmoid;
                break;
            case 0x94acb4aa: /* "Hardmax" */
                rop = r->op_Hardmax;
                break;
            case 0xdfd9b28f: /* "Identity" */
                rop = r->op_Identity;
                break;
            case 0x00597414: /* "If" */
                rop = r->op_If;
                break;
            case 0xfb0902c1: /* "InstanceNormalization" */
                rop = r->op_InstanceNormalization;
                break;
            case 0x0d68519e: /* "IsInf" */
                rop = r->op_IsInf;
                break;
            case 0x0d68651e: /* "IsNaN" */
                rop = r->op_IsNaN;
                break;
            case 0x0b880111: /* "LRN" */
                rop = r->op_LRN;
                break;
            case 0x7c882885: /* "LSTM" */
                rop = r->op_LSTM;
                break;
            case 0xea2c5c33: /* "LeakyRelu" */
                rop = r->op_LeakyRelu;
                break;
            case 0x7c88793c: /* "Less" */
                rop = r->op_Less;
                break;
            case 0x0b8804e7: /* "Log" */
                rop = r->op_Log;
                break;
            case 0x7c88a33f: /* "Loop" */
                rop = r->op_Loop;
                break;
            case 0x07f77ce8: /* "LpNormalization" */
                rop = r->op_LpNormalization;
                break;
            case 0xc13f923b: /* "LpPool" */
                rop = r->op_LpPool;
                break;
            case 0xc2987915: /* "MatMul" */
                rop = r->op_MatMul;
                break;
            case 0x62fbd803: /* "MatMulInteger" */
                rop = r->op_MatMulInteger;
                break;
            case 0x0b88076b: /* "Max" */
                rop = r->op_Max;
                break;
            case 0x15f18a25: /* "MaxPool" */
                rop = r->op_MaxPool;
                break;
            case 0x018c06cf: /* "MaxRoiPool" */
                rop = r->op_MaxRoiPool;
                break;
            case 0x641501e8: /* "MaxUnpool" */
                rop = r->op_MaxUnpool;
                break;
            case 0x7c890346: /* "Mean" */
                rop = r->op_Mean;
                break;
            case 0x0b880869: /* "Min" */
                rop = r->op_Min;
                break;
            case 0x0b880925: /* "Mod" */
                rop = r->op_Mod;
                break;
            case 0x0b8809f3: /* "Mul" */
                rop = r->op_Mul;
                break;
            case 0xaec55410: /* "Multinomial" */
                rop = r->op_Multinomial;
                break;
            case 0x0b880c1f: /* "Neg" */
                rop = r->op_Neg;
                break;
            case 0x254e25a1: /* "NonMaxSuppression" */
                rop = r->op_NonMaxSuppression;
                break;
            case 0x82e45c50: /* "NonZero" */
                rop = r->op_NonZero;
                break;
            case 0x0b880d76: /* "Not" */
                rop = r->op_Not;
                break;
            case 0xc825b932: /* "OneHot" */
                rop = r->op_OneHot;
                break;
            case 0x005974e6: /* "Or" */
                rop = r->op_Or;
                break;
            case 0x0dd55b8d: /* "PRelu" */
                rop = r->op_PRelu;
                break;
            case 0x0b88141a: /* "Pad" */
                rop = r->op_Pad;
                break;
            case 0x0b8815fb: /* "Pow" */
                rop = r->op_Pow;
                break;
            case 0xe569f427: /* "QLinearConv" */
                rop = r->op_QLinearConv;
                break;
            case 0xfe108481: /* "QLinearMatMul" */
                rop = r->op_QLinearMatMul;
                break;
            case 0x37138211: /* "QuantizeLinear" */
                rop = r->op_QuantizeLinear;
                break;
            case 0x0b881a13: /* "RNN" */
                rop = r->op_RNN;
                break;
            case 0xc100684f: /* "RandomNormal" */
                rop = r->op_RandomNormal;
                break;
            case 0xa0b57174: /* "RandomNormalLike" */
                rop = r->op_RandomNormalLike;
                break;
            case 0xf8e97c66: /* "RandomUniform" */
                rop = r->op_RandomUniform;
                break;
            case 0x10a8b90b: /* "RandomUniformLike" */
                rop = r->op_RandomUniformLike;
                break;
            case 0x73d06f69: /* "Reciprocal" */
                rop = r->op_Reciprocal;
                break;
            case 0x7944853a: /* "ReduceL1" */
                rop = r->op_ReduceL1;
                break;
            case 0x7944853b: /* "ReduceL2" */
                rop = r->op_ReduceL2;
                break;
            case 0xeab46d14: /* "ReduceLogSum" */
                rop = r->op_ReduceLogSum;
                break;
            case 0x9a057a01: /* "ReduceLogSumExp" */
                rop = r->op_ReduceLogSumExp;
                break;
            case 0xa1d53763: /* "ReduceMax" */
                rop = r->op_ReduceMax;
                break;
            case 0xdc7c323e: /* "ReduceMean" */
                rop = r->op_ReduceMean;
                break;
            case 0xa1d53861: /* "ReduceMin" */
                rop = r->op_ReduceMin;
                break;
            case 0xdc7e1072: /* "ReduceProd" */
                rop = r->op_ReduceProd;
                break;
            case 0xa1d55372: /* "ReduceSum" */
                rop = r->op_ReduceSum;
                break;
            case 0x20917223: /* "ReduceSumSquare" */
                rop = r->op_ReduceSumSquare;
                break;
            case 0x7c8bc29d: /* "Relu" */
                rop = r->op_Relu;
                break;
            case 0x9fdbcf8d: /* "Reshape" */
                rop = r->op_Reshape;
                break;
            case 0xce8a9197: /* "Resize" */
                rop = r->op_Resize;
                break;
            case 0x5d77301a: /* "ReverseSequence" */
                rop = r->op_ReverseSequence;
                break;
            case 0x830cb9da: /* "RoiAlign" */
                rop = r->op_RoiAlign;
                break;
            case 0x0e09b7cd: /* "Round" */
                rop = r->op_Round;
                break;
            case 0x7c8c450a: /* "Scan" */
                rop = r->op_Scan;
                break;
            case 0xe6ece5fb: /* "Scatter" */
                rop = r->op_Scatter;
                break;
            case 0xb4db6f18: /* "ScatterElements" */
                rop = r->op_ScatterElements;
                break;
            case 0x55be5b0d: /* "ScatterND" */
                rop = r->op_ScatterND;
                break;
            case 0x7c8c4efe: /* "Selu" */
                rop = r->op_Selu;
                break;
            case 0xe537ccd3: /* "SequenceAt" */
                rop = r->op_SequenceAt;
                break;
            case 0xa52772e3: /* "SequenceConstruct" */
                rop = r->op_SequenceConstruct;
                break;
            case 0x5e6e772d: /* "SequenceEmpty" */
                rop = r->op_SequenceEmpty;
                break;
            case 0x5e70f50e: /* "SequenceErase" */
                rop = r->op_SequenceErase;
                break;
            case 0x35a57cb3: /* "SequenceInsert" */
                rop = r->op_SequenceInsert;
                break;
            case 0x3bff64e0: /* "SequenceLength" */
                rop = r->op_SequenceLength;
                break;
            case 0x0e17a4d6: /* "Shape" */
                rop = r->op_Shape;
                break;
            case 0xd11575d4: /* "Shrink" */
                rop = r->op_Shrink;
                break;
            case 0xf5548151: /* "Sigmoid" */
                rop = r->op_Sigmoid;
                break;
            case 0x7c8c5f56: /* "Sign" */
                rop = r->op_Sign;
                break;
            case 0x0b8821ef: /* "Sin" */
                rop = r->op_Sin;
                break;
            case 0x7c8c6037: /* "Sinh" */
                rop = r->op_Sinh;
                break;
            case 0x7c8c61c0: /* "Size" */
                rop = r->op_Size;
                break;
            case 0x0e19f6b5: /* "Slice" */
                rop = r->op_Slice;
                break;
            case 0x6bec36a5: /* "Softplus" */
                rop = r->op_Softplus;
                break;
            case 0x6bedcd32: /* "Softsign" */
                rop = r->op_Softsign;
                break;
            case 0xa4436289: /* "SpaceToDepth" */
                rop = r->op_SpaceToDepth;
                break;
            case 0x0e1c35d1: /* "Split" */
                rop = r->op_Split;
                break;
            case 0x50e66fcd: /* "SplitToSequence" */
                rop = r->op_SplitToSequence;
                break;
            case 0x7c8c82cf: /* "Sqrt" */
                rop = r->op_Sqrt;
                break;
            case 0x08f69207: /* "Squeeze" */
                rop = r->op_Squeeze;
                break;
            case 0xf404645f: /* "StringNormalizer" */
                rop = r->op_StringNormalizer;
                break;
            case 0x0b88236f: /* "Sub" */
                rop = r->op_Sub;
                break;
            case 0x0b88237a: /* "Sum" */
                rop = r->op_Sum;
                break;
            case 0x0b882528: /* "Tan" */
                rop = r->op_Tan;
                break;
            case 0x7c8cca90: /* "Tanh" */
                rop = r->op_Tanh;
                break;
            case 0x46fbf3df: /* "TfIdfVectorizer" */
                rop = r->op_TfIdfVectorizer;
                break;
            case 0xa646ea33: /* "ThresholdedRelu" */
                rop = r->op_ThresholdedRelu;
                break;
            case 0x7c8cec53: /* "Tile" */
                rop = r->op_Tile;
                break;
            case 0x7c8d0643: /* "TopK" */
                rop = r->op_TopK;
                break;
            case 0x940b3944: /* "Transpose" */
                rop = r->op_Transpose;
                break;
            case 0xd6278d9c: /* "Unique" */
                rop = r->op_Unique;
                break;
            case 0xc836156a: /* "Unsqueeze" */
                rop = r->op_Unsqueeze;
                break;
            case 0xae63c66c: /* "Upsample" */
                rop = r->op_Upsample;
                break;
            case 0x0e601820: /* "Where" */
                rop = r->op_Where;
                break;
            case 0x0b8837fe: /* "Xor" */
                rop = r->op_Xor;
                break;

            case 0x7c8388ee: /* "Celu" */
                rop = r->op_Celu;
                break;
            case 0x718dbc56: /* "DynamicQuantizeLinear" */
                rop = r->op_DynamicQuantizeLinear;
                break;
            case 0x7b2541c8: /* "GreaterOrEqual" */
                rop = r->op_GreaterOrEqual;
                break;
            case 0x60d9a535: /* "LessOrEqual" */
                rop = r->op_LessOrEqual;
                break;
            case 0xf8c82769: /* "LogSoftmax" */
                rop = r->op_LogSoftmax;
                break;
            case 0xbb8f2396: /* "MeanVarianceNormalization" */
                rop = r->op_MeanVarianceNormalization;
                break;
            case 0x6ed111df: /* "NegativeLogLikelihoodLoss" */
                rop = r->op_NegativeLogLikelihoodLoss;
                break;
            case 0x0e01ebd2: /* "Range" */
                rop = r->op_Range;
                break;
            case 0x034529c7: /* "Softmax" */
                rop = r->op_Softmax;
                break;
            case 0x522154a3: /* "SoftmaxCrossEntropyLoss" */
                rop = r->op_SoftmaxCrossEntropyLoss;
                break;

            default:
                rop = NULL;
                break;
        }
        if(rop)
            rop(n);
    }
}



void * resolver_default_create(void)
{
	return NULL;
}

void resolver_default_destroy(void * rctx)
{
}

Resolver resolver_default = 
{
    .name 							= "default",

    .create							= resolver_default_create,
    .destroy						= resolver_default_destroy,

    #ifdef UONNX_OPS_ABS
    .op_Abs							= resolver_default_op_Abs,
    #endif

    #ifdef UONNX_OPS_ACOS
    .op_Acos						= resolver_default_op_Acos,
    #endif

    #ifdef UONNX_OPS_ACOSH
    .op_Acosh						= resolver_default_op_Acosh,
    #endif

    #ifdef UONNX_OPS_ADD
    .op_Add							= resolver_default_op_Add,
    #endif

    #ifdef UONNX_OPS_AND
    .op_And							= resolver_default_op_And,
    #endif

    #ifdef UONNX_OPS_ARGMAX
    .op_ArgMax						= resolver_default_op_ArgMax,
    #endif

    #ifdef UONNX_OPS_ARGMIN
    .op_ArgMin						= resolver_default_op_ArgMin,
    #endif

    #ifdef UONNX_OPS_ASIN
    .op_Asin						= resolver_default_op_Asin,
    #endif

    #ifdef UONNX_OPS_ASINH
    .op_Asinh						= resolver_default_op_Asinh,
    #endif

    #ifdef UONNX_OPS_ATAN
    .op_Atan						= resolver_default_op_Atan,
    #endif

    #ifdef UONNX_OPS_ATANH
    .op_Atanh						= resolver_default_op_Atanh,
    #endif

    #ifdef UONNX_OPS_AVERAGEPOOL
    .op_AveragePool					= resolver_default_op_AveragePool,
    #endif

    // .op_BatchNormalization			= resolver_default_op_BatchNormalization,
    // .op_BitShift					= resolver_default_op_BitShift,
    // .op_Cast						= resolver_default_op_Cast,

    #ifdef UONNX_OPS_CEIL
    .op_Ceil						= resolver_default_op_Ceil,
    #endif

    // .op_Clip						= resolver_default_op_Clip,
    // .op_Compress					= resolver_default_op_Compress,
    // .op_Concat						= resolver_default_op_Concat,
    // .op_ConcatFromSequence			= resolver_default_op_ConcatFromSequence,
    // .op_Constant					= resolver_default_op_Constant,
    // .op_ConstantOfShape				= resolver_default_op_ConstantOfShape,

    #ifdef UONNX_OPS_CONV
    .op_Conv						= resolver_default_op_Conv,
    #endif

    // .op_ConvInteger					= resolver_default_op_ConvInteger,
    // .op_ConvTranspose				= resolver_default_op_ConvTranspose,

    #ifdef UONNX_OPS_COS
    .op_Cos							= resolver_default_op_Cos,
    #endif

    #ifdef UONNX_OPS_COSH
    .op_Cosh						= resolver_default_op_Cosh,
    #endif

    // .op_CumSum						= resolver_default_op_CumSum,
    // .op_DepthToSpace				= resolver_default_op_DepthToSpace,
    // .op_DequantizeLinear			= resolver_default_op_DequantizeLinear,
    // .op_Det							= resolver_default_op_Det,
    // .op_Div							= resolver_default_op_Div,
    // .op_Dropout						= resolver_default_op_Dropout,
    // .op_Einsum						= resolver_default_op_Einsum,

    #ifdef UONNX_OPS_ELU
    .op_Elu							= resolver_default_op_Elu,
    #endif

    #ifdef UONNX_OPS_EQUAL
    .op_Equal						= resolver_default_op_Equal,
    #endif

    #ifdef UONNX_OPS_ERF
    .op_Erf							= resolver_default_op_Erf,
    #endif

    #ifdef UONNX_OPS_EXP
    .op_Exp							= resolver_default_op_Exp,
    #endif

    // .op_Expand						= resolver_default_op_Expand,
    // .op_EyeLike						= resolver_default_op_EyeLike,
    // .op_Flatten						= resolver_default_op_Flatten,

    #ifdef UONNX_OPS_FLOOR
    .op_Floor						= resolver_default_op_Floor,
    #endif

    // .op_GRU							= resolver_default_op_GRU,
    // .op_Gather						= resolver_default_op_Gather,
    // .op_GatherElements				= resolver_default_op_GatherElements,
    // .op_GatherND					= resolver_default_op_GatherND,
    #ifdef UONNX_OPS_GEMM
    .op_Gemm						= resolver_default_op_Gemm,
    #endif

    #ifdef UONNX_OPS_GLOBALAVERAGEPOOL
    .op_GlobalAveragePool			= resolver_default_op_GlobalAveragePool,
    #endif

    #ifdef UONNX_OPS_GLOBALLPPOOL
    .op_GlobalLpPool				= resolver_default_op_GlobalLpPool,
    #endif

    #ifdef UONNX_OPS_GLOBALMAXPOOL
    .op_GlobalMaxPool				= resolver_default_op_GlobalMaxPool,
    #endif

    #ifdef UONNX_OPS_GREATER
    .op_Greater						= resolver_default_op_Greater,
    #endif

    #ifdef UONNX_OPS_HARDSIGMOID
    .op_HardSigmoid					= resolver_default_op_HardSigmoid,
    #endif

    // .op_Hardmax						= resolver_default_op_Hardmax,
    // .op_Identity					= resolver_default_op_Identity,
    // .op_If							= resolver_default_op_If,
    // .op_InstanceNormalization		= resolver_default_op_InstanceNormalization,
    // .op_IsInf						= resolver_default_op_IsInf,
    // .op_IsNaN						= resolver_default_op_IsNaN,
    // .op_LRN							= resolver_default_op_LRN,
    // .op_LSTM						= resolver_default_op_LSTM,

    #ifdef UONNX_OPS_LEAKYRELU
    .op_LeakyRelu					= resolver_default_op_LeakyRelu,
    #endif

    #ifdef UONNX_OPS_LESS
    .op_Less						= resolver_default_op_Less,
    #endif

    #ifdef UONNX_OPS_LOG
    .op_Log							= resolver_default_op_Log,
    #endif

    // .op_Loop						= resolver_default_op_Loop,
    // .op_LpNormalization				= resolver_default_op_LpNormalization,
    // .op_LpPool						= resolver_default_op_LpPool,

    #ifdef UONNX_OPS_MATMUL
    .op_MatMul						= resolver_default_op_MatMul,
    #endif 

    // .op_MatMulInteger				= resolver_default_op_MatMulInteger,

    #ifdef UONNX_OPS_MAX
    .op_Max							= resolver_default_op_Max,
    #endif

    #ifdef UONNX_OPS_MAXPOOL
    .op_MaxPool						= resolver_default_op_MaxPool,
    #endif

    // .op_MaxRoiPool					= resolver_default_op_MaxRoiPool,
    // .op_MaxUnpool					= resolver_default_op_MaxUnpool,

    #ifdef UONNX_OPS_MEAN
    .op_Mean						= resolver_default_op_Mean,
    #endif

    #ifdef UONNX_OPS_MIN
    .op_Min							= resolver_default_op_Min,
    #endif
    
    #ifdef UONNX_OPS_MOD
    .op_Mod							= resolver_default_op_Mod,
    #endif

    #ifdef UONNX_OPS_MUL
    .op_Mul							= resolver_default_op_Mul,
    #endif

    // .op_Multinomial					= resolver_default_op_Multinomial,

    #ifdef UONNX_OPS_NEG
    .op_Neg							= resolver_default_op_Neg,
    #endif

    // .op_NonMaxSuppression			= resolver_default_op_NonMaxSuppression,
    // .op_NonZero						= resolver_default_op_NonZero,
    // .op_Not							= resolver_default_op_Not,
    // .op_OneHot						= resolver_default_op_OneHot,
    // .op_Or							= resolver_default_op_Or,

    #ifdef UONNX_OPS_PRELU
    .op_PRelu						= resolver_default_op_PRelu,
    #endif

    // .op_Pad							= resolver_default_op_Pad,
    
    #ifdef UONNX_OPS_POW
    .op_Pow							= resolver_default_op_Pow,
    #endif

    // .op_QLinearConv					= resolver_default_op_QLinearConv,
    // .op_QLinearMatMul				= resolver_default_op_QLinearMatMul,
    // .op_QuantizeLinear				= resolver_default_op_QuantizeLinear,
    // .op_RNN							= resolver_default_op_RNN,
    // .op_RandomNormal				= resolver_default_op_RandomNormal,
    // .op_RandomNormalLike			= resolver_default_op_RandomNormalLike,
    // .op_RandomUniform				= resolver_default_op_RandomUniform,
    // .op_RandomUniformLike			= resolver_default_op_RandomUniformLike,
    // .op_Reciprocal					= resolver_default_op_Reciprocal,
    // .op_ReduceL1					= resolver_default_op_ReduceL1,
    // .op_ReduceL2					= resolver_default_op_ReduceL2,
    // .op_ReduceLogSum				= resolver_default_op_ReduceLogSum,
    // .op_ReduceLogSumExp				= resolver_default_op_ReduceLogSumExp,
    // .op_ReduceMax					= resolver_default_op_ReduceMax,
    // .op_ReduceMean					= resolver_default_op_ReduceMean,
    // .op_ReduceMin					= resolver_default_op_ReduceMin,
    // .op_ReduceProd					= resolver_default_op_ReduceProd,
    // .op_ReduceSum					= resolver_default_op_ReduceSum,
    // .op_ReduceSumSquare				= resolver_default_op_ReduceSumSquare,

    #ifdef UONNX_OPS_RELU
    .op_Relu						= resolver_default_op_Relu,
    #endif

    #ifdef UONNX_OPS_RESHAPE
    .op_Reshape						= resolver_default_op_Reshape,
    #endif

    // .op_Resize						= resolver_default_op_Resize,
    // .op_ReverseSequence				= resolver_default_op_ReverseSequence,
    // .op_RoiAlign					= resolver_default_op_RoiAlign,
    // .op_Round						= resolver_default_op_Round,
    // .op_Scan						= resolver_default_op_Scan,
    // .op_Scatter						= resolver_default_op_Scatter,
    // .op_ScatterElements				= resolver_default_op_ScatterElements,
    // .op_ScatterND					= resolver_default_op_ScatterND,

    #ifdef UONNX_OPS_SELU
    .op_Selu						= resolver_default_op_Selu,
    #endif

    // .op_SequenceAt					= resolver_default_op_SequenceAt,
    // .op_SequenceConstruct			= resolver_default_op_SequenceConstruct,
    // .op_SequenceEmpty				= resolver_default_op_SequenceEmpty,
    // .op_SequenceErase				= resolver_default_op_SequenceErase,
    // .op_SequenceInsert				= resolver_default_op_SequenceInsert,
    // .op_SequenceLength				= resolver_default_op_SequenceLength,
    // .op_Shape						= resolver_default_op_Shape,
    // .op_Shrink						= resolver_default_op_Shrink,

    #ifdef UONNX_OPS_SIGMOID
    .op_Sigmoid						= resolver_default_op_Sigmoid,
    #endif

    #ifdef UONNX_OPS_SIGN
    .op_Sign						= resolver_default_op_Sign,
    #endif

    #ifdef UONNX_OPS_SIN
    .op_Sin							= resolver_default_op_Sin,
    #endif

    #ifdef UONNX_OPS_SINH
    .op_Sinh						= resolver_default_op_Sinh,
    #endif

    // .op_Size						= resolver_default_op_Size,
    // .op_Slice						= resolver_default_op_Slice,
    // .op_Softplus					= resolver_default_op_Softplus,
    // .op_Softsign					= resolver_default_op_Softsign,
    // .op_SpaceToDepth				= resolver_default_op_SpaceToDepth,
    // .op_Split						= resolver_default_op_Split,
    // .op_SplitToSequence				= resolver_default_op_SplitToSequence,
    // .op_Sqrt						= resolver_default_op_Sqrt,
    // .op_Squeeze						= resolver_default_op_Squeeze,
    // .op_StringNormalizer			= resolver_default_op_StringNormalizer,

    #ifdef UONNX_OPS_SUB
    .op_Sub							= resolver_default_op_Sub,
    #endif

    #ifdef UONNX_OPS_SUM
    .op_Sum							= resolver_default_op_Sum,
    #endif

    #ifdef UONNX_OPS_TAN
    .op_Tan							= resolver_default_op_Tan,
    #endif

    #ifdef UONNX_OPS_TANH
    .op_Tanh						= resolver_default_op_Tanh,
    #endif

    // .op_TfIdfVectorizer				= resolver_default_op_TfIdfVectorizer,
    // .op_ThresholdedRelu				= resolver_default_op_ThresholdedRelu,
    // .op_Tile						= resolver_default_op_Tile,
    // .op_TopK						= resolver_default_op_TopK,
    
    #ifdef UONNX_OPS_TRANSPOSE
    .op_Transpose					= resolver_default_op_Transpose,
    #endif

    // .op_Trilu						= resolver_default_op_Trilu,
    // .op_Unique						= resolver_default_op_Unique,
    // .op_Unsqueeze					= resolver_default_op_Unsqueeze,
    // .op_Upsample					= resolver_default_op_Upsample,
    // .op_Where						= resolver_default_op_Where,
    // .op_Xor							= resolver_default_op_Xor,

    #ifdef UONNX_OPS_CELU
    .op_Celu						= resolver_default_op_Celu,
    #endif

    // .op_DynamicQuantizeLinear		= resolver_default_op_DynamicQuantizeLinear,

    #ifdef UONNX_OPS_GREATEROREQUAL
    .op_GreaterOrEqual				= resolver_default_op_GreaterOrEqual,
    #endif

    // .op_HardSwish					= resolver_default_op_HardSwish,

    #ifdef UONNX_OPS_LESSOREQUAL
    .op_LessOrEqual					= resolver_default_op_LessOrEqual,
    #endif

    #ifdef UONNX_OPS_LOGSOFTMAX
    .op_LogSoftmax					= resolver_default_op_LogSoftmax,
    #endif
    // .op_MeanVarianceNormalization	= resolver_default_op_MeanVarianceNormalization,
    // .op_NegativeLogLikelihoodLoss	= resolver_default_op_NegativeLogLikelihoodLoss,
    // .op_Range						= resolver_default_op_Range,

    #ifdef UONNX_OPS_SOFTMAX
    .op_Softmax						= resolver_default_op_Softmax,
    #endif
    // .op_SoftmaxCrossEntropyLoss		= resolver_default_op_SoftmaxCrossEntropyLoss,
};