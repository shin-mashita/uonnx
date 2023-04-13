#include <uonnx.h>

int main()
{
	ModelProto * model = NULL;
	GraphProto * graph = NULL;
	
	const char * filename = "./scratch/model.onnx";

	model = load_model(filename);
	graph = model->graph;
	

	for(int i=0; i < model->graph->n_node; i++)
	{
		printf("Node %d: %s, Opset: %s\n",
								i, 	
								model->graph->node[i]->name, 
								model->graph->node[i]->op_type);

		printf("\tInput tensors (%d)\n", model->graph->node[i]->n_input);
		for(int j=0; j < model->graph->node[i]->n_input; j++)
		{
			printf("\t\tInput %d: %s\n", j, model->graph->node[i]->input[j]);
		}

		printf("\tOutput tensors (%d)\n", model->graph->node[i]->n_output);
		for(int j=0; j < model->graph->node[i]->n_output; j++)
		{
			printf("\t\tInput %d: %s\n", j, model->graph->node[i]->output[j]);
		}
		for(int j=0; j < model->graph->node[i]->n_attribute; j++)
		{
			printf("\tAttr %d: %s\n", j, model->graph->node[i]->attribute[j]->name);
		}
	}

	for(int i = 0; i < model->graph->n_initializer; i++)
	{
		printf("Initializer %d: %s", i, model->graph->initializer[i]->name);
		printf(", DType: %s", TensorType2String(model->graph->initializer[i]->data_type));
		switch(model->graph->initializer[i]->data_type)
		{
			case TENSOR_TYPE_FLOAT32:
				printf(", NumDatas: %d: \n", model->graph->initializer[i]->n_float_data);
				break;
			case TENSOR_TYPE_INT64:
				printf(", NumDatas: %d: \n", model->graph->initializer[i]->n_int64_data);
				break;
			default: // TODO: Log for other dtypes
				break;
		}
		

		// if (model->graph->initializer[i]->n_float_data > 50)
		// {
		// 	for(int j=1; j < 50+1; j++)
		// 	{
		// 		if(!j%5)printf("%f \n\t\t", model->graph->initializer[i]->float_data[j]);
		// 		else printf("%f ", model->graph->initializer[i]->float_data[j]);
		// 	}
		// }

	}


	free_model(model);

    return 0;
}