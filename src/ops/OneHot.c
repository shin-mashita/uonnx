#include <uonnx.h>

void resolver_default_op_OneHot(struct onnx_node_t * n)
{
	if(n->opset >= 11)
	{
	}
	else if(n->opset >= 9)
	{
	}
}
