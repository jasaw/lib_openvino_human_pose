CFLAGS = -W -Wall -pthread -g -O3 $(EXTRA_CFLAGS)
RM = rm -rf
CC ?= $(CROSS)gcc
SRCDIR=.

CFLAGS += -I$(SRCDIR) -I$(SRCDIR)/..
#CFLAGS += -Wl,-rpath -Wl,$(DETECTION_LIB_PATH)
LDFLAGS = -ldl -ljpeg

POSE_SRC:=$(wildcard $(SRCDIR)/*.c)
POSE_OBJ:=$(POSE_SRC:%.c=%.o)

.DEFAULT_GOAL := all

all: human_pose_estomator_test

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

human_pose_estomator_test: $(POSE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	-@$(RM) human_pose_estomator_test $(POSE_OBJ)
