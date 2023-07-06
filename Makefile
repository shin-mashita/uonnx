# CC and FLAGS
CC			:= gcc
AR			:= ar
# CFLAGS		:= -std=c99 -g3 -ggdb -Wall -O3
CFLAGS		:= -g3 -ggdb -Wall -O3
LDFLAGS		:= -g


APP			?= examples/benchmarks/cpu/mnist
APPDIRS 	:= $(sort $(dir $(wildcard ./$(APP)/*/)))
APPFILES	:= $(foreach dir, $(APPDIRS), $(wildcard $(dir)*.c))
APPOBJS		:= $(foreach file, $(APPFILES:.c=.o), $(subst ./, ./build/, $(file)))

SRCDIRS		+= ./src/ ./src/proto/ 
OPSDIRS		:= ./src/ops/

BUILDDIR	+= ./build
INCDIRS		+= -I ./src/

LIBNAME		:= libuonnx.a
LIBPATH		:= $(BUILDDIR)/lib/$(LIBNAME)
LIBFLAGS	:= -L $(BUILDDIR)/lib/
LIBS		:= -lm

OPS			?= $(foreach dir, $(OPSDIRS), $(wildcard $(dir)*.c))

SRCS		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)*.c))
SRCS		+= $(OPS)

COBJS		:= $(foreach file, $(SRCS:.c=.o), $(subst ./, ./build/, $(file)))
OBJS		:= $(COBJS) $(APPOBJS)

MODEL		?= examples/benchmarks/cpu/mnist/mnist.onnx
PLANNER		?= 

.PHONY: all lib run benchmark run_with_lib clean test make_test

all: clean lib

# TODO: ifneq def for Makefile for ops stripping

# Compile C files into C objects 

$(APPOBJS): $(BUILDDIR)/%.o:%.c
	@mkdir -p $(dir $@)
	@echo [CC] APP: $^
	@$(CC) -o $@ -c $(CFLAGS) $(INCDIRS) $^

$(COBJS): $(BUILDDIR)/%.o:%.c
	@mkdir -p $(dir $@)
	@echo [CC] SRC: $^
	@$(CC) -o $@ -c $(CFLAGS) $(INCDIRS) $^


# Rule for library
lib: $(LIBPATH)
$(LIBPATH):$(COBJS)
	@mkdir -p $(dir $@)
	@echo [AR] Archiving $(LIBNAME)
	@$(AR) -rcs $@ $(COBJS)
	@echo [MK] Library path is $(LIBPATH)

# Rule for making apps
run: $(COBJS) $(APPOBJS)
	@echo [LD] Linking $(APP)
	@$(CC) -o $(BUILDDIR)/$(APP).out $(OBJS) $(CFLAGS) $(INCDIRS) $(LIBS) -static
	@echo
	@echo [MK] App executable found at $(BUILDDIR)/$(APP).out
	@echo

$(eval mnist:;@:)
$(eval kws:;@:)
$(eval vww:;@:)

benchmark:
ifeq (mnist, $(filter mnist,$(MAKECMDGOALS)))
	@clear
	@echo [MK] Compiling mnist benchmark...
	@make prep MODEL=examples/benchmarks/cpu/mnist/mnist.onnx
	@make run APP=examples/benchmarks/cpu/mnist
	@echo [MK] Running mnist benchmark on CPU...
	@./build/examples/benchmarks/cpu/mnist.out
	@make clean
else ifeq (kws, $(filter kws,$(MAKECMDGOALS)))
	@clear
	@echo [MK] Compiling kws benchmark...
	@make prep MODEL=examples/benchmarks/cpu/kws/kws_float32_9.onnx
	@make run APP=examples/benchmarks/cpu/kws
	@echo [MK] Running kws benchmark on CPU...
	@./build/examples/benchmarks/cpu/kws.out
	@make clean
else ifeq (vww, $(filter vww,$(MAKECMDGOALS)))
	@clear
	@echo [MK] Compiling vww benchmark...
	@make prep MODEL=examples/benchmarks/cpu/vww/vww_float32_9.onnx
	@make run APP=examples/benchmarks/cpu/vww
	@echo [MK] Running vww benchmark on CPU...
	@./build/examples/benchmarks/cpu/vww.out
	@make clean
else
	@clear
	@echo [MK] Compiling reference benchmark...
	@make prep MODEL=examples/benchmarks/cpu/ref/reference.onnx
	@make run APP=examples/benchmarks/cpu/ref
	@echo [MK] Running ref benchmark on CPU...
	@./build/examples/benchmarks/cpu/ref.out
	@make clean
endif

prep:
	@python tools/preprocessor.py --model ./${MODEL}

# Rule for linking app with lib
run_with_lib: lib $(APPOBJS)
	@echo [LD] Linking $(APP)
	@$(CC) -static -o $(BUILDDIR)/$(APP).out $(LIBFLAGS) $(APPOBJS) -luonnx $(LIBS)
	@echo
	@echo [MK] App executable found at $(BUILDDIR)/$(APP).out 
	@echo

# Rule for cleaning ./build
clean:
	@rm -rf $(BUILDDIR)
	@python tools/preprocessor.py --restore

# Used for debugging
make_test:
	@echo BUILDDIR: $(BUILDDIR)
	@echo SRCDIRS: $(SRCDIRS)
	@echo OPSDIRS: $(OPSDIRS)

	@echo OPS: $(OPS)
	@echo SRCS: $(SRCS)
	@echo COBJS: $(COBJS)

	@echo APP: $(APPDIRS)
	@echo APPFILES: $(APPFILES)
	@echo APPOBJS: $(APPOBJS)
	
