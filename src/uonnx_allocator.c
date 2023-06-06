#include "uonnx_allocator.h"

Tensor *tensor_init(const char *name,
                    TensorType type,
                    int *dims,
                    int ndim,
                    TensorArena *arena,
                    Tensor *tensor,
                    int data_idx)
{
    int i = 0;
    size_t ndata = 1;
    Tensor *t;

    if (arena)
    {
        t = tensor;
    }
    else
    {
        t = malloc(sizeof(Tensor)); // TODO: Find way to free if arena == NULL
    }

    t->name = strdup(name);
    if (!t->name)
    {
        if (!arena)
            free(t);
        return NULL;
    }

    t->strides = malloc(sizeof(int) * ndim);
    t->dims = malloc(sizeof(int) * ndim);

    if ((!t->strides) || (!t->dims))
    {
        if (t->strides)
            free(t->strides);
        if (t->dims)
            free(t->dims);

        free(t->name);
        if (!arena)
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
        if (!arena)
            free(t);
        return NULL;
    }

    if (arena)
    {
        t->datas = arena->datas + data_idx;
    }
    else
    {
        t->datas = malloc(onnx_tensor_type_sizeof(t->type) * ndata);
        if (!t->datas)
        {
            free(t->strides);
            free(t->dims);
            free(t->name);
            free(t);
            return NULL;
        }
        memset(t->datas, 0, onnx_tensor_type_sizeof(t->type) * ndata);
    }

    t->type = type;
    t->isInitializer = 0;
    t->ndim = ndim;

    return t;
}

Tensor *tensor_init_from_value_info(ValueInfoProto *v, TensorArena *arena, Tensor *tensor, int data_idx)
{
    Tensor *t;
    int type;
    int *dims = NULL;
    int ndim;
    int i;

    if (!v || !v->name)
    {
        return NULL;
    }

    if (v->type->value_case == ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE)
    {
        type = v->type->tensor_type->elem_type;
        ndim = v->type->tensor_type->shape->n_dim;

        if (ndim > 0)
        {
            dims = malloc(sizeof(int) * ndim);
            if (dims)
            {
                for (i = 0; i < ndim; i++)
                {
                    if (v->type->tensor_type->shape->dim[i]->value_case == ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
                    {
                        dims[i] = v->type->tensor_type->shape->dim[i]->dim_value;
                    }
                    else
                    {
                        dims[i] = 1;
                    }
                }
            }
            else
            {
                return NULL;
            }
        }

        t = tensor_init(v->name, type, dims, ndim, arena, tensor, data_idx);

        if (dims)
        {
            free(dims);
            dims = NULL;
        }

        return t;
    }

    return NULL;
}


Tensor *tensor_init_from_proto(TensorProto *tp, Tensor *tensor)
{
    int i, n;
    size_t ndata = 1;
    Tensor *t;

    if (tensor == NULL)
    {
        t = malloc(sizeof(Tensor));
    }
    else
    {
        t = tensor;
    }

    if (tp)
    {
        t->name = strdup(tp->name);
        t->type = tp->data_type;

        t->dims = malloc(sizeof(int) * tp->n_dims);
        t->strides = malloc(sizeof(int) * tp->n_dims);

        if (t->dims && t->strides)
        {
            t->strides[tp->n_dims - 1] = 1;

            for (i = tp->n_dims - 2; i >= 0; i--)
            {
                t->strides[i] = tp->dims[i + 1] * t->strides[i + 1];
            }

            for (i = 0; i < tp->n_dims; i++)
            {
                t->dims[i] = tp->dims[i];
            }

            t->ndim = tp->n_dims;
        }

        for (i = 0; i < t->ndim; i++)
        {
            ndata *= t->dims[i];
        }

        t->isInitializer = 1;

        switch (t->type)
        {
        case TENSOR_TYPE_FLOAT32:
        {
            t->datas = tp->float_data;
            t->ndata = tp->n_float_data;
        }
            break;


        case TENSOR_TYPE_INT64:
            t->datas = tp->int64_data;
            t->ndata = tp->n_int64_data;
            break;

        case TENSOR_TYPE_FLOAT16:
        {
            // TODO: Allocation handling scheme for float16 tensors
            t->datas = tp->int32_data;
            t->ndata = ndata;
        }

            break;

        default:
            break; // TODO: ADD SUPPORT FOR OTHER DTYPES
        }

        if (ndata != t->ndata)
        {
            return NULL;
        }

        return t;
    }

    return NULL;
}

void free_tensor(Tensor *t)
{
    if (t)
    {
        if (t->name)
            free(t->name);
        if (t->dims)
            free(t->dims);
        if (t->strides)
            free(t->strides);
        if (t->datas)
            free(t->datas);
        free(t);
    }
}

void free_tensor_from_arena(Tensor *t)
{
    if (t)
    {
        if (t->name)
            free(t->name);
        if (t->dims)
            free(t->dims);
        if (t->strides)
            free(t->strides);
        free(t);
    }
}

void tensor_apply(void *datas, size_t size, Tensor *t)
{
    if (t->name != NULL && t)
    {
        memcpy(t->datas, datas, size); // Run with sizeof()
    }
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

Graph *graph_init(GraphProto *gproto, ModelProto *model, TensorArena *arena, Planner *planner)
{
    int i = 0, j = 0, isInitializer = 0;
    int n_tensor_count = 0;
    char *p, *domain;
    Graph *g;
    Node *n;
    ValueInfoProto *vp;

    if (!gproto)
        return NULL;

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

    // Add inputs to graph
    for (i = 0; i < gproto->n_input; i++)
    {
        // Add initializers into arena
        vp = gproto->input[i];
        if (!tensor_search(arena, vp->name))
        {
            for (j = 0; j < gproto->n_initializer; j++)
            {
                if (strcmp(gproto->initializer[j]->name, vp->name) == 0)
                {
                    if (n_tensor_count >= arena->n_tensors)
                    {
                        // printf("Tensor blocks not enough. Expand arena.\n"); TODO: Conditional Compile
                        free(g->nodes);
                        return NULL;
                    }
                    tensor_init_from_proto(gproto->initializer[j], arena->tensors[n_tensor_count]);
                    n_tensor_count++;
                    isInitializer = 1;
                }
            }

            // Initialize non-initializer inputs
            if (!isInitializer)
            {
                if (n_tensor_count >= arena->n_tensors)
                {
                    // printf("Tensor blocks not enough. Expand arena.\n"); TODO: Conditional Compile
                    free(g->nodes);
                    return NULL;
                }
                tensor_init_from_value_info(vp, arena, arena->tensors[n_tensor_count], get_plan(vp->name, planner));
                n_tensor_count++;
            }

            isInitializer = 0;
        }
    }

    // Add outputs to graph
    for (i = 0; i < gproto->n_output; i++)
    {
        vp = gproto->output[i];
        if (!tensor_search(arena, vp->name))
        {
            if (n_tensor_count >= arena->n_tensors)
            {
                // printf("Tensor blocks not enough. Expand arena.\n"); TODO: Conditional Compile
                free(g->nodes);
                return NULL;
            }
            tensor_init_from_value_info(vp, arena, arena->tensors[n_tensor_count], get_plan(vp->name, planner));
            n_tensor_count++;
        }
    }

    for (i = 0; i < gproto->n_value_info; i++)
    {
        vp = gproto->value_info[i];
        if (!tensor_search(arena, vp->name))
        {
            if (n_tensor_count >= arena->n_tensors)
            {
                // printf("Tensor blocks not enough. Expand arena.\n"); TODO: Conditional Compile
                free(g->nodes);
                return NULL;
            }
            tensor_init_from_value_info(vp, arena, arena->tensors[n_tensor_count], get_plan(vp->name, planner));
            n_tensor_count++;
        }
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

        resolver_solve_operator(&resolver_default, n, n->proto->op_type);

        if (!n->reshape)
            n->reshape = reshape_dummy;
        if (!n->operator)
            n->operator= operator_dummy;
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
