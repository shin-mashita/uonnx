#include "uonnx_loader.h"

ModelProto * load_model_buf(const void * buf, size_t len)
{
    ModelProto * model = onnx__model_proto__unpack(NULL, len, buf);
    return model;
}

ModelProto * load_model(const char * filename)
{
    ModelProto * model;
    FILE * fp;
	void * buf;
	size_t l, len;

    fp = fopen(filename, "rb");

    if(fp)
    {
        fseek(fp, 0L, SEEK_END);
		l = ftell(fp);
		fseek(fp, 0L, SEEK_SET);
		if(l > 0)
        {
            buf = malloc(l);
            if(buf)
            {
                for(len = 0; len < l; len += fread(buf + len, 1, l - len, fp));
                model = load_model_buf(buf, len);
                free(buf);
            }
        }
        fclose(fp);
    }

    return model;
}

void free_model(ModelProto * model)
{
    onnx__model_proto__free_unpacked(model, NULL);
}