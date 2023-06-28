#include "uonnx_allocator.h"

// Tensor * tensor_alloc //withdatas for no arena application

Tensor * tensor_alloc_nodatas(uint32_t tensor_id, TensorType type, int * dims, int ndim, uint8_t isInitializer)
{
    int i = 0;
    size_t ndata = 1;
    Tensor * t;

    t = malloc(sizeof(Tensor));
    if(!t)
    {
        return NULL;
    }

    t->id = tensor_id;
    t->isInitializer = isInitializer;
    t->datas = NULL;
    t->type = type;
    t->ndim = ndim;

    if(!isInitializer)
    {
        t->strides = malloc(sizeof(int)*ndim);
        t->dims = malloc(sizeof(int) * ndim);

        if((!t->strides)||(!t->dims))
        {
            if(t->strides)
                free(t->strides);
            if(t->dims)
                free(t->dims);
            free(t);
            return NULL;
        }

        t->strides[ndim - 1] = 1;
        for (i = ndim - 2; i >= 0; i--)
        {
            t->strides[i] = dims[i + 1] * t->strides[i + 1];
        }

        memcpy(t->dims, dims, sizeof(int) * ndim);

        for (i = 0; i < ndim; i++)
            ndata *= t->dims[i];

        if (ndata != 0)
            t->ndata = ndata;
        else
        {
            free(t->strides);
            free(t->dims);
            free(t->name);
            free(t);
            return NULL;
        }
    }
    else
    {
        t->dims = NULL;
        t->strides = NULL;
        t->ndata = 0;
    }

    return t;
}

static int reshape_dummy(Node *n)
{
    return 1;
}

static void operator_dummy(Node *n)
{
    // Conditional compile
    // printf("\033[45;37mUnsupported opset\033[0m => %s-%d (%s)\r\n", n->proto->op_type, n->opset, (strlen(n->proto->domain) > 0) ? n->proto->domain : "ai.onnx");
    return;
}

Graph * graph_init_from_PlannerProto(ModelProto * model, PlannerProto * planner, TensorArena * arena)
{
    int i = 0, j = 0;
    char *p, *domain;
    Graph *g;
    Node *n;

    if(!model || !planner || !arena)
    {
        return NULL;
    }

    GraphProto * gproto = model->graph;

    g = malloc(sizeof(Graph));
    if (!g)
        return NULL;

    memset(g, 0, sizeof(Graph));

    g->nodes = malloc(sizeof(Node) * gproto->n_node);
    if (!g->nodes)
    {
        free(g);
        return NULL;
    }

    g->nlen = gproto->n_node;

    for (i = 0; i < gproto->n_initializer; i++)
    {
        arena_add_initializer(gproto->initializer[i], arena);
    }

    for (i = 0; i < planner->n_plans; i++)
    {
        arena_add_intermediate(planner->plans[i], arena);
    }

    for (i = 0; i < g->nlen; i++)
    {
        n = &g->nodes[i];
        n->proto = gproto->node[i];

        domain = n->proto->domain;
        if (!domain || (strlen(domain) == 0))
            domain = "ai.onnx";
        for (j = 0; j < model->n_opset_import; j++)
        {
            p = model->opset_import[j]->domain;
            if (!p || (strlen(p) == 0))
                p = "ai.onnx";
            if (strcmp(domain, p) == 0)
            {
                n->opset = model->opset_import[j]->version;
                break;
            }
        }

        n->inputs = malloc(sizeof(Tensor *) * gproto->node[i]->n_input);
        if (n->inputs)
        {
            n->ninputs = gproto->node[i]->n_input;
            for (j = 0; j < gproto->node[i]->n_input; j++)
            {
                n->inputs[j] = tensor_search(arena, gproto->node[i]->input[j]);
            }
        }

        n->outputs = malloc(sizeof(Tensor *) * gproto->node[i]->n_output);

        if (n->outputs)
        {
            n->noutputs = gproto->node[i]->n_output;
            for (j = 0; j < gproto->node[i]->n_output; j++)
            {
                n->outputs[j] = tensor_search(arena, gproto->node[i]->output[j]);
            }
        }

        resolver_solve_operator(&resolver_default, n);

        if (!n->reshape)
            n->reshape = reshape_dummy;
        if (!n->op)
            n->op= operator_dummy;
        if (n->init)
        {
            if (n->init(n) <= 0)
            {
                if (g->nodes)
                {
                    for (j = 0; j <= i; j++)
                    {
                        n = &g->nodes[j];
                        if (n->exit)
                            n->exit(n);
                        if (n->inputs)
                            free(n->inputs);
                        if (n->outputs)
                            free(n->outputs);
                    }
                    free(g->nodes);
                }
                free(g);
                return NULL;
            }
        }

        if (n->reshape)
            n->reshape(n);
    }

    return g;
}

void tensor_apply(void *datas, size_t size, Tensor *t)
{
    if (t)
    {
        memcpy(t->datas, datas, size); // Run with sizeof()
    }
}

void free_tensor_from_arena(Tensor *t)
{
    if (t)
    {
        if (t->dims)
            free(t->dims);
        if (t->strides)
            free(t->strides);
        free(t);
    }
}

void free_graph(Graph *g)
{
    Node *n;
    int i = 0;

    if (g)
    {
        if (g->nodes)
        {
            for (i = 0; i < g->nlen; i++)
            {
                n = &g->nodes[i];
                if (n->exit)
                    n->exit(n);
                if (n->inputs)
                    free(n->inputs);
                if (n->outputs)
                    free(n->outputs);
            }
            free(g->nodes);
        }
        free(g);
    }
}
