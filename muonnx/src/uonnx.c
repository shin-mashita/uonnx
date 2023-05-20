#include <uonnx.h>

Context * uonnx_init(   const char * filename, 
                        const void * model_buf,
                        size_t model_len,
                        void * input_buf, 
                        size_t input_len,
                        char * input_name, 
                        void * output_buf, 
                        size_t output_len,
                        char * output_name, 
                        Planner * planner)
{
    if(!planner)
    {
        return NULL; // TODO: Add generate_planner_FFD here
    }

    Context * context = (Context *)malloc(sizeof(Context));
    if(!context)
        return NULL;

    if(filename)
    {
        context->model = load_model(filename);
    }
    else if(model_buf && model_len > 0)
    {
        context->model = load_model_buf(model_buf, model_len);
    }
    else
    {
        free(context);
        return NULL;
    }

    if(!context->model)
    {
        free(context);
        return NULL;
    }

    context->arena = arena_init_from_planner(planner);
    if(!context->arena)
    {
        free_model(context->model);
        free(context);
        return NULL;
    }

    context->graph = graph_init(context->model->graph, context->model, context->arena, planner);
    if(!context->graph)
    {
        free_arena(context->arena);
        free_model(context->model);
        free(context);
        return NULL;
    }

    context->input_buf = input_buf;
    context->input_len = input_len;
    context->input_tensor = tensor_search(context->arena, input_name);

    context->output_buf = output_buf;
    context->output_len = output_len;
    context->output_tensor = tensor_search(context->arena, output_name);

    context->planner = planner;
    
    return context;
}

void uonnx_run(Context * context)
{
    Node * n;
    int i = 0;
    Graph * g = context->graph;

    tensor_apply((void*)(context->input_buf), context->input_len, context->input_tensor);
    for(i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];
        n->operator(n);
    }

    if(context->output_buf && context->output_len > 0)
    {
        memcpy(context->output_buf, context->output_tensor->datas, context->output_len);
    }
}

void uonnx_free(Context * context)
{
    if(context)
    {
        free_graph(context->graph);
        free_arena(context->arena);
        free_model(context->model);
        free_planner(context->planner);
        free(context);
    }
}
