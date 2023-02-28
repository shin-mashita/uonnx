# TODO: Create makefile that run all builds %.o and %.s on dir ./build
# FOR MAKEFILE PREPROCESSING: CHECK ONNX_INCLUDE by cONNXr

CC			:= gcc
CFLAGS		:= -g -ggdb -Wall -O3


SRCDIRS		+= $(dir $(wildcard ./src/*/))
BUILDDIR	+= ./build
INCDIRS		+= -I ./src
LIBDIRS		:= -L ./src
LIBS		:= -lm

CFILES		:= $(foreach dir, $(SRCDIRS), $(wildcard $(dir)*.c))

COBJS		:= $(foreach file, $(CFILES:.c=.o), $(subst ./, ./build/, $(file)))
OBJS		:= $(COBJS)

SCRATCHFILE	:= ./src/scratch.c

# Compile and create all object files to directory ./build
$(BUILDDIR)/%.o:%.c
	@mkdir -p $(dir $@)
	@echo [CC] $^
	@$(CC) -o $@ -c $(CFLAGS) $^

.PHONY: test scratch clean

#
scratch: $(BUILDDIR)/scratch clean
$(BUILDDIR)/scratch: $(OBJS)
	@echo [LD] Linking $@
	@$(CC) $(LIBDIRS) $^ -o $@ $(LIBS) -static
	@echo [RR] Running $@
	@echo
	@./$@
	@echo

clean:
	@rm -rf $(BUILDDIR)

test: scratch clean
	

