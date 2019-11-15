CXXFLAGS = -W -Wall -pthread -g -O3 -march=armv7-a -DNDEBUG $(EXTRA_CXXFLAGS)
RM = rm -rf
CXX ?= $(CROSS)g++
AR ?= $(CROSS)ar
PREFIX?=/usr
SRCDIR=.
INTEL_OPENVINO_DIR?=/opt/intel/openvino
OPENCV_DIR?=$(INTEL_OPENVINO_DIR)/opencv
SYSTEM_TYPE:=$(shell ls $(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/lib)
IE_PLUGINS_PATH?=$(INTEL_OPENVINO_DIR)/inference_engine/lib/$(SYSTEM_TYPE)
FFMPEG_LIB_DIR?=/usr/local/lib

CXXFLAGS += -I$(SRCDIR)
CXXFLAGS += -I$(INTEL_OPENVINO_DIR)/deployment_tools/inference_engine/include
CXXFLAGS += -I$(OPENCV_DIR)/include
CXXFLAGS += -Wl,-rpath -Wl,$(IE_PLUGINS_PATH)
CXXFLAGS += -Wl,-rpath -Wl,$(OPENCV_DIR)/lib
CXXFLAGS += -Wl,-rpath -Wl,$(FFMPEG_LIB_DIR)
LDFLAGS += -L$(IE_PLUGINS_PATH)
LDFLAGS += -L$(OPENCV_DIR)/lib
LDFLAGS += -L$(FFMPEG_LIB_DIR)
LDFLAGS += -linference_engine
LDFLAGS += -lcpu_extension -lHeteroPlugin -lmyriadPlugin
LDFLAGS += -lopencv_core
LDFLAGS += -lopencv_imgcodecs -lopencv_imgproc
#LDFLAGS += -lopencv_stitching -lopencv_photo
#LDFLAGS += -lopencv_calib3d -lopencv_features2d
#LDFLAGS += -lopencv_flann -lopencv_gapi -lopencv_highgui
#LDFLAGS += -lopencv_ml -lopencv_dnn -lopencv_objdetect
#LDFLAGS += -lopencv_video -lopencv_videoio
#LDFLAGS += -lopencv_videoio_ffmpeg -lopencv_videoio_gstreamer
LDFLAGS += -lswscale -lavutil

POSE_CPP:=$(wildcard $(SRCDIR)/*.cpp)
POSE_OBJ:=$(POSE_CPP:%.cpp=%.o)

.DEFAULT_GOAL := all

all: libopenvinohumanpose.a libopenvinohumanpose.so

%.o: %.cpp
	$(CXX) -fPIC -c -o $@ $< $(CXXFLAGS)

libopenvinohumanpose.a: $(POSE_OBJ)
	$(AR) rcs $@ $^

libopenvinohumanpose.so: $(POSE_OBJ)
	$(CXX) -shared -fPIC $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) *.a *.so $(POSE_OBJ)

install:
	mkdir -p $(PREFIX)/include/libopenvinohumanpose/
	install -D -m 0755 $(SRCDIR)/*.h $(PREFIX)/include/libopenvinohumanpose/
	install -D -m 0755 *.a $(PREFIX)/lib
	install -D -m 0755 *.so $(PREFIX)/lib
