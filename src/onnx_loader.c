#include "onnx_loader.h"

Onnx__ModelProto * onnx_load_model(const void * buf, size_t len)
{
    Onnx__ModelProto * model = onnx__model_proto__unpack(NULL, len, buf);
    return model;
}

Onnx__ModelProto * onnx_load_model_from_file(const char * filename)
{
    Onnx__ModelProto * model;
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
                model = onnx_load_model(buf, len);
            }
        }
    }

    return model;
}
