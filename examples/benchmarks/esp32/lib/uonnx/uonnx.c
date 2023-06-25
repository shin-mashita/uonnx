#include <uonnx.h>

Context * uonnx_init(const void * model_buf, size_t model_len, const void * planner_buf, size_t planner_len)
{
    Context * ctx = (Context *)malloc(sizeof(Context));
    
    if(!ctx)
        return NULL;

    ctx->model = load_model_buf(model_buf, model_len);
    ctx->planner = load_planner_buf(planner_buf, planner_len);
    
    if(!ctx->model || !ctx->planner)
    {
        if(ctx->model)
            free_model(ctx->model);
        if(ctx->planner)
            free_plannerproto(ctx->planner);
        free(ctx);
        return NULL;
    }

    ctx->arena = arena_init(ctx->planner->arena->max_ntensors, ctx->planner->arena->max_bytes);

    if(!ctx->arena)
    {
        free_model(ctx->model);
        free_plannerproto(ctx->planner);
        free(ctx);
        return NULL;
    }

    ctx->graph = graph_init_from_PlannerProto(ctx->model, ctx->planner, ctx->arena);
    
    if(!ctx->graph)
    {
        free_arena(ctx->arena);
        free_model(ctx->model);
        free_plannerproto(ctx->planner);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void uonnx_run(Context * ctx)
{
    Node * n;
    int i = 0;

    for(i = 0; i < ctx->graph->nlen; i++)
    {
        n = &ctx->graph->nodes[i];
        n->op(n);
    }
}

void uonnx_free(Context * ctx)
{
    if(ctx)
    {
        free_graph(ctx->graph);
        free_arena(ctx->arena);
        free_model(ctx->model);
        free_plannerproto(ctx->planner);
        free(ctx);
    }
}
