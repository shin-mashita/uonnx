#include <uonnx.h>

void onnx_run(struct onnx_context_t * ctx)
{
	struct onnx_node_t * n;
	int i;

	if(ctx)
	{
		for(i = 0; i < ctx->g->nlen; i++)
		{
			n = &ctx->g->nodes[i];
			if(n->reshape(n))
				n->operator(n);
		}
	}
}
