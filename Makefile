################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
DEBUG = 1
CXX = g++
# Compiler flags
CFLAGS = -Wall -std=c++11
OBJ_DIR := build

ifeq ($(DEBUG), 1)
	CFLAGS += -g -DDEBUG_FLAG
endif
CUDA_VER:=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-test3-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=6.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= $(wildcard *.c) $(wildcard *.cpp) /opt/nvidia/deepstream/deepstream-6.0/sources/apps/apps-common/src/deepstream_perf.c  

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-6.0/sources/includes/ \
		-I /usr/local/cuda-$(CUDA_VER)/include -I/opt/nvidia/deepstream/deepstream-6.0/sources/apps/apps-common/includes

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)


# build object files
%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
$(APP): $(OBJS) Makefile
	$(CXX) -g -o $(APP) $(OBJS) $(CFLAGS) $(LIBS) 

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)
