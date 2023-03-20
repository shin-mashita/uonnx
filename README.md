# uONNX
Memory-efficient ONNX inference engine for tinyML


## Working `make` commands 

* `make all`: rebuild static library 
* `make lib`: build static lib
* `make run APP=your_app_name`: create executable for app using manual linking of src objs. By default, `APP=scratch`.
* `make run_with_lib APP=your_app_name`: create executable by linking app to static lib. 
* `make clean`: remove ./build
* `make test_scratch`: compile src and run scratch app. 
* `make make_test`: just some debugger for makefile.


onnx_loader.h
onnx_allocator.h