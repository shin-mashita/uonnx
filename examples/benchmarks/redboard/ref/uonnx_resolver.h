
#ifndef __UONNX_RESOLVER_H__
#define __UONNX_RESOLVER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "uonnx.h"

// Add conditional compilation if defines for each ops UONNX_OPS_ABS

typedef struct Resolver {
    const char * name;

    void * (*create)(void);
    void (*destroy)(void * rctx);

    void (*op_Abs)(Node * n);
    void (*op_Acos)(Node * n);
    void (*op_Acosh)(Node * n);
    void (*op_Add)(Node * n);
    void (*op_And)(Node * n);
    void (*op_ArgMax)(Node * n);
    void (*op_ArgMin)(Node * n);
    void (*op_Asin)(Node * n);
    void (*op_Asinh)(Node * n);
    void (*op_Atan)(Node * n);
    void (*op_Atanh)(Node * n);
    void (*op_AveragePool)(Node * n);
    void (*op_BatchNormalization)(Node * n);
    void (*op_BitShift)(Node * n);
    void (*op_Cast)(Node * n);
    void (*op_Ceil)(Node * n);
    void (*op_Clip)(Node * n);
    void (*op_Compress)(Node * n);
    void (*op_Concat)(Node * n);
    void (*op_ConcatFromSequence)(Node * n);
    void (*op_Constant)(Node * n);
    void (*op_ConstantOfShape)(Node * n);
    void (*op_Conv)(Node * n);
    void (*op_ConvInteger)(Node * n);
    void (*op_ConvTranspose)(Node * n);
    void (*op_Cos)(Node * n);
    void (*op_Cosh)(Node * n);
    void (*op_CumSum)(Node * n);
    void (*op_DepthToSpace)(Node * n);
    void (*op_DequantizeLinear)(Node * n);
    void (*op_Det)(Node * n);
    void (*op_Div)(Node * n);
    void (*op_Dropout)(Node * n);
    void (*op_Einsum)(Node * n);
    void (*op_Elu)(Node * n);
    void (*op_Equal)(Node * n);
    void (*op_Erf)(Node * n);
    void (*op_Exp)(Node * n);
    void (*op_Expand)(Node * n);
    void (*op_EyeLike)(Node * n);
    void (*op_Flatten)(Node * n);
    void (*op_Floor)(Node * n);
    void (*op_GRU)(Node * n);
    void (*op_Gather)(Node * n);
    void (*op_GatherElements)(Node * n);
    void (*op_GatherND)(Node * n);
    void (*op_Gemm)(Node * n);
    void (*op_GlobalAveragePool)(Node * n);
    void (*op_GlobalLpPool)(Node * n);
    void (*op_GlobalMaxPool)(Node * n);
    void (*op_Greater)(Node * n);
    void (*op_HardSigmoid)(Node * n);
    void (*op_Hardmax)(Node * n);
    void (*op_Identity)(Node * n);
    void (*op_If)(Node * n);
    void (*op_InstanceNormalization)(Node * n);
    void (*op_IsInf)(Node * n);
    void (*op_IsNaN)(Node * n);
    void (*op_LRN)(Node * n);
    void (*op_LSTM)(Node * n);
    void (*op_LeakyRelu)(Node * n);
    void (*op_Less)(Node * n);
    void (*op_Log)(Node * n);
    void (*op_Loop)(Node * n);
    void (*op_LpNormalization)(Node * n);
    void (*op_LpPool)(Node * n);
    void (*op_MatMul)(Node * n);
    void (*op_MatMulInteger)(Node * n);
    void (*op_Max)(Node * n);
    void (*op_MaxPool)(Node * n);
    void (*op_MaxRoiPool)(Node * n);
    void (*op_MaxUnpool)(Node * n);
    void (*op_Mean)(Node * n);
    void (*op_Min)(Node * n);
    void (*op_Mod)(Node * n);
    void (*op_Mul)(Node * n);
    void (*op_Multinomial)(Node * n);
    void (*op_Neg)(Node * n);
    void (*op_NonMaxSuppression)(Node * n);
    void (*op_NonZero)(Node * n);
    void (*op_Not)(Node * n);
    void (*op_OneHot)(Node * n);
    void (*op_Or)(Node * n);
    void (*op_PRelu)(Node * n);
    void (*op_Pad)(Node * n);
    void (*op_Pow)(Node * n);
    void (*op_QLinearConv)(Node * n);
    void (*op_QLinearMatMul)(Node * n);
    void (*op_QuantizeLinear)(Node * n);
    void (*op_RNN)(Node * n);
    void (*op_RandomNormal)(Node * n);
    void (*op_RandomNormalLike)(Node * n);
    void (*op_RandomUniform)(Node * n);
    void (*op_RandomUniformLike)(Node * n);
    void (*op_Reciprocal)(Node * n);
    void (*op_ReduceL1)(Node * n);
    void (*op_ReduceL2)(Node * n);
    void (*op_ReduceLogSum)(Node * n);
    void (*op_ReduceLogSumExp)(Node * n);
    void (*op_ReduceMax)(Node * n);
    void (*op_ReduceMean)(Node * n);
    void (*op_ReduceMin)(Node * n);
    void (*op_ReduceProd)(Node * n);
    void (*op_ReduceSum)(Node * n);
    void (*op_ReduceSumSquare)(Node * n);
    void (*op_Relu)(Node * n);
    void (*op_Reshape)(Node * n);
    void (*op_Resize)(Node * n);
    void (*op_ReverseSequence)(Node * n);
    void (*op_RoiAlign)(Node * n);
    void (*op_Round)(Node * n);
    void (*op_Scan)(Node * n);
    void (*op_Scatter)(Node * n);
    void (*op_ScatterElements)(Node * n);
    void (*op_ScatterND)(Node * n);
    void (*op_Selu)(Node * n);
    void (*op_SequenceAt)(Node * n);
    void (*op_SequenceConstruct)(Node * n);
    void (*op_SequenceEmpty)(Node * n);
    void (*op_SequenceErase)(Node * n);
    void (*op_SequenceInsert)(Node * n);
    void (*op_SequenceLength)(Node * n);
    void (*op_Shape)(Node * n);
    void (*op_Shrink)(Node * n);
    void (*op_Sigmoid)(Node * n);
    void (*op_Sign)(Node * n);
    void (*op_Sin)(Node * n);
    void (*op_Sinh)(Node * n);
    void (*op_Size)(Node * n);
    void (*op_Slice)(Node * n);
    void (*op_Softplus)(Node * n);
    void (*op_Softsign)(Node * n);
    void (*op_SpaceToDepth)(Node * n);
    void (*op_Split)(Node * n);
    void (*op_SplitToSequence)(Node * n);
    void (*op_Sqrt)(Node * n);
    void (*op_Squeeze)(Node * n);
    void (*op_StringNormalizer)(Node * n);
    void (*op_Sub)(Node * n);
    void (*op_Sum)(Node * n);
    void (*op_Tan)(Node * n);
    void (*op_Tanh)(Node * n);
    void (*op_TfIdfVectorizer)(Node * n);
    void (*op_ThresholdedRelu)(Node * n);
    void (*op_Tile)(Node * n);
    void (*op_TopK)(Node * n);
    void (*op_Transpose)(Node * n);
    void (*op_Trilu)(Node * n);
    void (*op_Unique)(Node * n);
    void (*op_Unsqueeze)(Node * n);
    void (*op_Upsample)(Node * n);
    void (*op_Where)(Node * n);
    void (*op_Xor)(Node * n);

    void (*op_Celu)(Node * n);
    void (*op_DynamicQuantizeLinear)(Node * n);
    void (*op_GreaterOrEqual)(Node * n);
    void (*op_HardSwish)(Node * n);
    void (*op_LessOrEqual)(Node * n);
    void (*op_LogSoftmax)(Node * n);
    void (*op_MeanVarianceNormalization)(Node * n);
    void (*op_NegativeLogLikelihoodLoss)(Node * n);
    void (*op_Range)(Node * n);
    void (*op_Softmax)(Node * n);
    void (*op_SoftmaxCrossEntropyLoss)(Node * n);
} Resolver;

void resolver_solve_operator(Resolver * r, Node * n);

void * resolver_default_create(void);
void resolver_default_destroy(void * rctx);

void resolver_default_op_Abs(Node * n);
void resolver_default_op_Acos(Node * n);
void resolver_default_op_Acosh(Node * n);
void resolver_default_op_Add(Node * n);
void resolver_default_op_And(Node * n);
void resolver_default_op_ArgMax(Node * n);
void resolver_default_op_ArgMin(Node * n);
void resolver_default_op_Asin(Node * n);
void resolver_default_op_Asinh(Node * n);
void resolver_default_op_Atan(Node * n);
void resolver_default_op_Atanh(Node * n);
void resolver_default_op_AveragePool(Node * n);
void resolver_default_op_BatchNormalization(Node * n);
void resolver_default_op_BitShift(Node * n);
void resolver_default_op_Cast(Node * n);
void resolver_default_op_Ceil(Node * n);
void resolver_default_op_Clip(Node * n);
void resolver_default_op_Compress(Node * n);
void resolver_default_op_Concat(Node * n);
void resolver_default_op_ConcatFromSequence(Node * n);
void resolver_default_op_Constant(Node * n);
void resolver_default_op_ConstantOfShape(Node * n);
void resolver_default_op_Conv(Node * n);
void resolver_default_op_ConvInteger(Node * n);
void resolver_default_op_ConvTranspose(Node * n);
void resolver_default_op_Cos(Node * n);
void resolver_default_op_Cosh(Node * n);
void resolver_default_op_CumSum(Node * n);
void resolver_default_op_DepthToSpace(Node * n);
void resolver_default_op_DequantizeLinear(Node * n);
void resolver_default_op_Det(Node * n);
void resolver_default_op_Div(Node * n);
void resolver_default_op_Dropout(Node * n);
void resolver_default_op_Einsum(Node * n);
void resolver_default_op_Elu(Node * n);
void resolver_default_op_Equal(Node * n);
void resolver_default_op_Erf(Node * n);
void resolver_default_op_Exp(Node * n);
void resolver_default_op_Expand(Node * n);
void resolver_default_op_EyeLike(Node * n);
void resolver_default_op_Flatten(Node * n);
void resolver_default_op_Floor(Node * n);
void resolver_default_op_GRU(Node * n);
void resolver_default_op_Gather(Node * n);
void resolver_default_op_GatherElements(Node * n);
void resolver_default_op_GatherND(Node * n);
void resolver_default_op_Gemm(Node * n);
void resolver_default_op_GlobalAveragePool(Node * n);
void resolver_default_op_GlobalLpPool(Node * n);
void resolver_default_op_GlobalMaxPool(Node * n);
void resolver_default_op_Greater(Node * n);
void resolver_default_op_HardSigmoid(Node * n);
void resolver_default_op_Hardmax(Node * n);
void resolver_default_op_Identity(Node * n);
void resolver_default_op_If(Node * n);
void resolver_default_op_InstanceNormalization(Node * n);
void resolver_default_op_IsInf(Node * n);
void resolver_default_op_IsNaN(Node * n);
void resolver_default_op_LRN(Node * n);
void resolver_default_op_LSTM(Node * n);
void resolver_default_op_LeakyRelu(Node * n);
void resolver_default_op_Less(Node * n);
void resolver_default_op_Log(Node * n);
void resolver_default_op_Loop(Node * n);
void resolver_default_op_LpNormalization(Node * n);
void resolver_default_op_LpPool(Node * n);
void resolver_default_op_MatMul(Node * n);
void resolver_default_op_MatMulInteger(Node * n);
void resolver_default_op_Max(Node * n);
void resolver_default_op_MaxPool(Node * n);
void resolver_default_op_MaxRoiPool(Node * n);
void resolver_default_op_MaxUnpool(Node * n);
void resolver_default_op_Mean(Node * n);
void resolver_default_op_Min(Node * n);
void resolver_default_op_Mod(Node * n);
void resolver_default_op_Mul(Node * n);
void resolver_default_op_Multinomial(Node * n);
void resolver_default_op_Neg(Node * n);
void resolver_default_op_NonMaxSuppression(Node * n);
void resolver_default_op_NonZero(Node * n);
void resolver_default_op_Not(Node * n);
void resolver_default_op_OneHot(Node * n);
void resolver_default_op_Or(Node * n);
void resolver_default_op_PRelu(Node * n);
void resolver_default_op_Pad(Node * n);
void resolver_default_op_Pow(Node * n);
void resolver_default_op_QLinearConv(Node * n);
void resolver_default_op_QLinearMatMul(Node * n);
void resolver_default_op_QuantizeLinear(Node * n);
void resolver_default_op_RNN(Node * n);
void resolver_default_op_RandomNormal(Node * n);
void resolver_default_op_RandomNormalLike(Node * n);
void resolver_default_op_RandomUniform(Node * n);
void resolver_default_op_RandomUniformLike(Node * n);
void resolver_default_op_Reciprocal(Node * n);
void resolver_default_op_ReduceL1(Node * n);
void resolver_default_op_ReduceL2(Node * n);
void resolver_default_op_ReduceLogSum(Node * n);
void resolver_default_op_ReduceLogSumExp(Node * n);
void resolver_default_op_ReduceMax(Node * n);
void resolver_default_op_ReduceMean(Node * n);
void resolver_default_op_ReduceMin(Node * n);
void resolver_default_op_ReduceProd(Node * n);
void resolver_default_op_ReduceSum(Node * n);
void resolver_default_op_ReduceSumSquare(Node * n);
void resolver_default_op_Relu(Node * n);
void resolver_default_op_Reshape(Node * n);
void resolver_default_op_Resize(Node * n);
void resolver_default_op_ReverseSequence(Node * n);
void resolver_default_op_RoiAlign(Node * n);
void resolver_default_op_Round(Node * n);
void resolver_default_op_Scan(Node * n);
void resolver_default_op_Scatter(Node * n);
void resolver_default_op_ScatterElements(Node * n);
void resolver_default_op_ScatterND(Node * n);
void resolver_default_op_Selu(Node * n);
void resolver_default_op_SequenceAt(Node * n);
void resolver_default_op_SequenceConstruct(Node * n);
void resolver_default_op_SequenceEmpty(Node * n);
void resolver_default_op_SequenceErase(Node * n);
void resolver_default_op_SequenceInsert(Node * n);
void resolver_default_op_SequenceLength(Node * n);
void resolver_default_op_Shape(Node * n);
void resolver_default_op_Shrink(Node * n);
void resolver_default_op_Sigmoid(Node * n);
void resolver_default_op_Sign(Node * n);
void resolver_default_op_Sin(Node * n);
void resolver_default_op_Sinh(Node * n);
void resolver_default_op_Size(Node * n);
void resolver_default_op_Slice(Node * n);
void resolver_default_op_Softplus(Node * n);
void resolver_default_op_Softsign(Node * n);
void resolver_default_op_SpaceToDepth(Node * n);
void resolver_default_op_Split(Node * n);
void resolver_default_op_SplitToSequence(Node * n);
void resolver_default_op_Sqrt(Node * n);
void resolver_default_op_Squeeze(Node * n);
void resolver_default_op_StringNormalizer(Node * n);
void resolver_default_op_Sub(Node * n);
void resolver_default_op_Sum(Node * n);
void resolver_default_op_Tan(Node * n);
void resolver_default_op_Tanh(Node * n);
void resolver_default_op_TfIdfVectorizer(Node * n);
void resolver_default_op_ThresholdedRelu(Node * n);
void resolver_default_op_Tile(Node * n);
void resolver_default_op_TopK(Node * n);
void resolver_default_op_Transpose(Node * n);
void resolver_default_op_Trilu(Node * n);
void resolver_default_op_Unique(Node * n);
void resolver_default_op_Unsqueeze(Node * n);
void resolver_default_op_Upsample(Node * n);
void resolver_default_op_Where(Node * n);
void resolver_default_op_Xor(Node * n);

void resolver_default_op_Celu(Node * n);
void resolver_default_op_DynamicQuantizeLinear(Node * n);
void resolver_default_op_GreaterOrEqual(Node * n);
void resolver_default_op_HardSwish(Node * n);
void resolver_default_op_LessOrEqual(Node * n);
void resolver_default_op_LogSoftmax(Node * n);
void resolver_default_op_MeanVarianceNormalization(Node * n);
void resolver_default_op_NegativeLogLikelihoodLoss(Node * n);
void resolver_default_op_Range(Node * n);
void resolver_default_op_Softmax(Node * n);
void resolver_default_op_SoftmaxCrossEntropyLoss(Node * n);

extern Resolver resolver_default;

#ifdef __cplusplus
}
#endif
#endif