
#ifndef __ONNX_RESOLVER_H__
#define __ONNX_RESOLVER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "onnx_config.h"
#include "onnx_dtypes.h"

struct onnx_resolver_t {
	const char * name;

	void * (*create)(void);
	void (*destroy)(void * rctx);

	void (*op_Abs)(struct onnx_node_t * n);
	void (*op_Acos)(struct onnx_node_t * n);
	void (*op_Acosh)(struct onnx_node_t * n);
	void (*op_Add)(struct onnx_node_t * n);
	void (*op_And)(struct onnx_node_t * n);
	void (*op_ArgMax)(struct onnx_node_t * n);
	void (*op_ArgMin)(struct onnx_node_t * n);
	void (*op_Asin)(struct onnx_node_t * n);
	void (*op_Asinh)(struct onnx_node_t * n);
	void (*op_Atan)(struct onnx_node_t * n);
	void (*op_Atanh)(struct onnx_node_t * n);
	void (*op_AveragePool)(struct onnx_node_t * n);
	void (*op_BatchNormalization)(struct onnx_node_t * n);
	void (*op_BitShift)(struct onnx_node_t * n);
	void (*op_Cast)(struct onnx_node_t * n);
	void (*op_Ceil)(struct onnx_node_t * n);
	void (*op_Clip)(struct onnx_node_t * n);
	void (*op_Compress)(struct onnx_node_t * n);
	void (*op_Concat)(struct onnx_node_t * n);
	void (*op_ConcatFromSequence)(struct onnx_node_t * n);
	void (*op_Constant)(struct onnx_node_t * n);
	void (*op_ConstantOfShape)(struct onnx_node_t * n);
	void (*op_Conv)(struct onnx_node_t * n);
	void (*op_ConvInteger)(struct onnx_node_t * n);
	void (*op_ConvTranspose)(struct onnx_node_t * n);
	void (*op_Cos)(struct onnx_node_t * n);
	void (*op_Cosh)(struct onnx_node_t * n);
	void (*op_CumSum)(struct onnx_node_t * n);
	void (*op_DepthToSpace)(struct onnx_node_t * n);
	void (*op_DequantizeLinear)(struct onnx_node_t * n);
	void (*op_Det)(struct onnx_node_t * n);
	void (*op_Div)(struct onnx_node_t * n);
	void (*op_Dropout)(struct onnx_node_t * n);
	void (*op_Einsum)(struct onnx_node_t * n);
	void (*op_Elu)(struct onnx_node_t * n);
	void (*op_Equal)(struct onnx_node_t * n);
	void (*op_Erf)(struct onnx_node_t * n);
	void (*op_Exp)(struct onnx_node_t * n);
	void (*op_Expand)(struct onnx_node_t * n);
	void (*op_EyeLike)(struct onnx_node_t * n);
	void (*op_Flatten)(struct onnx_node_t * n);
	void (*op_Floor)(struct onnx_node_t * n);
	void (*op_GRU)(struct onnx_node_t * n);
	void (*op_Gather)(struct onnx_node_t * n);
	void (*op_GatherElements)(struct onnx_node_t * n);
	void (*op_GatherND)(struct onnx_node_t * n);
	void (*op_Gemm)(struct onnx_node_t * n);
	void (*op_GlobalAveragePool)(struct onnx_node_t * n);
	void (*op_GlobalLpPool)(struct onnx_node_t * n);
	void (*op_GlobalMaxPool)(struct onnx_node_t * n);
	void (*op_Greater)(struct onnx_node_t * n);
	void (*op_HardSigmoid)(struct onnx_node_t * n);
	void (*op_Hardmax)(struct onnx_node_t * n);
	void (*op_Identity)(struct onnx_node_t * n);
	void (*op_If)(struct onnx_node_t * n);
	void (*op_InstanceNormalization)(struct onnx_node_t * n);
	void (*op_IsInf)(struct onnx_node_t * n);
	void (*op_IsNaN)(struct onnx_node_t * n);
	void (*op_LRN)(struct onnx_node_t * n);
	void (*op_LSTM)(struct onnx_node_t * n);
	void (*op_LeakyRelu)(struct onnx_node_t * n);
	void (*op_Less)(struct onnx_node_t * n);
	void (*op_Log)(struct onnx_node_t * n);
	void (*op_Loop)(struct onnx_node_t * n);
	void (*op_LpNormalization)(struct onnx_node_t * n);
	void (*op_LpPool)(struct onnx_node_t * n);
	void (*op_MatMul)(struct onnx_node_t * n);
	void (*op_MatMulInteger)(struct onnx_node_t * n);
	void (*op_Max)(struct onnx_node_t * n);
	void (*op_MaxPool)(struct onnx_node_t * n);
	void (*op_MaxRoiPool)(struct onnx_node_t * n);
	void (*op_MaxUnpool)(struct onnx_node_t * n);
	void (*op_Mean)(struct onnx_node_t * n);
	void (*op_Min)(struct onnx_node_t * n);
	void (*op_Mod)(struct onnx_node_t * n);
	void (*op_Mul)(struct onnx_node_t * n);
	void (*op_Multinomial)(struct onnx_node_t * n);
	void (*op_Neg)(struct onnx_node_t * n);
	void (*op_NonMaxSuppression)(struct onnx_node_t * n);
	void (*op_NonZero)(struct onnx_node_t * n);
	void (*op_Not)(struct onnx_node_t * n);
	void (*op_OneHot)(struct onnx_node_t * n);
	void (*op_Or)(struct onnx_node_t * n);
	void (*op_PRelu)(struct onnx_node_t * n);
	void (*op_Pad)(struct onnx_node_t * n);
	void (*op_Pow)(struct onnx_node_t * n);
	void (*op_QLinearConv)(struct onnx_node_t * n);
	void (*op_QLinearMatMul)(struct onnx_node_t * n);
	void (*op_QuantizeLinear)(struct onnx_node_t * n);
	void (*op_RNN)(struct onnx_node_t * n);
	void (*op_RandomNormal)(struct onnx_node_t * n);
	void (*op_RandomNormalLike)(struct onnx_node_t * n);
	void (*op_RandomUniform)(struct onnx_node_t * n);
	void (*op_RandomUniformLike)(struct onnx_node_t * n);
	void (*op_Reciprocal)(struct onnx_node_t * n);
	void (*op_ReduceL1)(struct onnx_node_t * n);
	void (*op_ReduceL2)(struct onnx_node_t * n);
	void (*op_ReduceLogSum)(struct onnx_node_t * n);
	void (*op_ReduceLogSumExp)(struct onnx_node_t * n);
	void (*op_ReduceMax)(struct onnx_node_t * n);
	void (*op_ReduceMean)(struct onnx_node_t * n);
	void (*op_ReduceMin)(struct onnx_node_t * n);
	void (*op_ReduceProd)(struct onnx_node_t * n);
	void (*op_ReduceSum)(struct onnx_node_t * n);
	void (*op_ReduceSumSquare)(struct onnx_node_t * n);
	void (*op_Relu)(struct onnx_node_t * n);
	void (*op_Reshape)(struct onnx_node_t * n);
	void (*op_Resize)(struct onnx_node_t * n);
	void (*op_ReverseSequence)(struct onnx_node_t * n);
	void (*op_RoiAlign)(struct onnx_node_t * n);
	void (*op_Round)(struct onnx_node_t * n);
	void (*op_Scan)(struct onnx_node_t * n);
	void (*op_Scatter)(struct onnx_node_t * n);
	void (*op_ScatterElements)(struct onnx_node_t * n);
	void (*op_ScatterND)(struct onnx_node_t * n);
	void (*op_Selu)(struct onnx_node_t * n);
	void (*op_SequenceAt)(struct onnx_node_t * n);
	void (*op_SequenceConstruct)(struct onnx_node_t * n);
	void (*op_SequenceEmpty)(struct onnx_node_t * n);
	void (*op_SequenceErase)(struct onnx_node_t * n);
	void (*op_SequenceInsert)(struct onnx_node_t * n);
	void (*op_SequenceLength)(struct onnx_node_t * n);
	void (*op_Shape)(struct onnx_node_t * n);
	void (*op_Shrink)(struct onnx_node_t * n);
	void (*op_Sigmoid)(struct onnx_node_t * n);
	void (*op_Sign)(struct onnx_node_t * n);
	void (*op_Sin)(struct onnx_node_t * n);
	void (*op_Sinh)(struct onnx_node_t * n);
	void (*op_Size)(struct onnx_node_t * n);
	void (*op_Slice)(struct onnx_node_t * n);
	void (*op_Softplus)(struct onnx_node_t * n);
	void (*op_Softsign)(struct onnx_node_t * n);
	void (*op_SpaceToDepth)(struct onnx_node_t * n);
	void (*op_Split)(struct onnx_node_t * n);
	void (*op_SplitToSequence)(struct onnx_node_t * n);
	void (*op_Sqrt)(struct onnx_node_t * n);
	void (*op_Squeeze)(struct onnx_node_t * n);
	void (*op_StringNormalizer)(struct onnx_node_t * n);
	void (*op_Sub)(struct onnx_node_t * n);
	void (*op_Sum)(struct onnx_node_t * n);
	void (*op_Tan)(struct onnx_node_t * n);
	void (*op_Tanh)(struct onnx_node_t * n);
	void (*op_TfIdfVectorizer)(struct onnx_node_t * n);
	void (*op_ThresholdedRelu)(struct onnx_node_t * n);
	void (*op_Tile)(struct onnx_node_t * n);
	void (*op_TopK)(struct onnx_node_t * n);
	void (*op_Transpose)(struct onnx_node_t * n);
	void (*op_Trilu)(struct onnx_node_t * n);
	void (*op_Unique)(struct onnx_node_t * n);
	void (*op_Unsqueeze)(struct onnx_node_t * n);
	void (*op_Upsample)(struct onnx_node_t * n);
	void (*op_Where)(struct onnx_node_t * n);
	void (*op_Xor)(struct onnx_node_t * n);

	void (*op_Celu)(struct onnx_node_t * n);
	void (*op_DynamicQuantizeLinear)(struct onnx_node_t * n);
	void (*op_GreaterOrEqual)(struct onnx_node_t * n);
	void (*op_HardSwish)(struct onnx_node_t * n);
	void (*op_LessOrEqual)(struct onnx_node_t * n);
	void (*op_LogSoftmax)(struct onnx_node_t * n);
	void (*op_MeanVarianceNormalization)(struct onnx_node_t * n);
	void (*op_NegativeLogLikelihoodLoss)(struct onnx_node_t * n);
	void (*op_Range)(struct onnx_node_t * n);
	void (*op_Softmax)(struct onnx_node_t * n);
	void (*op_SoftmaxCrossEntropyLoss)(struct onnx_node_t * n);
};

void resolver_solve_operator(struct onnx_resolver_t * r, struct onnx_node_t * n);

#ifdef __cplusplus
}
#endif
#endif