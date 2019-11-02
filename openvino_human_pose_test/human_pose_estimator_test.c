/*
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include "jpeg_reader.h"
#include "alt_detect.h"


typedef struct
{
    void *handle;
    int (*alt_detect_init)(const char *);
    void (*alt_detect_uninit)();
    int (*alt_detect_process)(unsigned char *, int, int);
    int (*alt_detect_result_ready)(void);
    int (*alt_detect_get_result)(alt_detect_result_t *);
    void (*alt_detect_free_result)(alt_detect_result_t *);
} lib_detect_info;


static int lib_detect_load_sym(void **func, void *handle, char *symbol)
{
    char *sym_error;

    *func = dlsym(handle, symbol);
    if ((sym_error = dlerror()) != NULL)
    {
        fprintf(stderr, "%s\n", sym_error);
        return -1;
    }
    return 0;
}


static int lib_detect_load(lib_detect_info *libdetect, const char *lib_detect_path)
{
    int err = 0;

    printf("Loading library: %s\n", lib_detect_path);

    libdetect->handle = dlopen(lib_detect_path, RTLD_LAZY);
    if (!libdetect->handle)
    {
        fprintf(stderr, "%s\n", dlerror());
        return -1;
    }

    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_init),   libdetect->handle, "alt_detect_init");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_uninit), libdetect->handle, "alt_detect_uninit");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_process), libdetect->handle, "alt_detect_process");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_result_ready), libdetect->handle, "alt_detect_result_ready");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_get_result), libdetect->handle, "alt_detect_get_result");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_free_result), libdetect->handle, "alt_detect_free_result");
    if (err)
        return -1;

    return 0;
}


static void lib_detect_unload(lib_detect_info *libdetect)
{
    if (libdetect->handle)
    {
        dlclose(libdetect->handle);
        libdetect->handle = NULL;
    }
}


static void syntax(const char *progname)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "%s [options]\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, " -l [libpath]              Detection library path\n");
    fprintf(stderr, " -c [configfile]           Detection library configuration file\n");
    fprintf(stderr, " -i [imagepath]            Input JPEG image path\n");
    fprintf(stderr, " -h                        Display this help page\n");
    fprintf(stderr, "\n");
    //fprintf(stderr, "Environment Variables:\n");
    //fprintf(stderr, "OPENVINO_MODEL_XML         Detection model XML path\n");
    //fprintf(stderr, "OPENVINO_MODEL_BIN         Detection model BIN path\n");
    //fprintf(stderr, "\n");
    //fprintf(stderr, "Optional Environment Variables:\n");
    //fprintf(stderr, "OPENVINO_TARGET_DEVICE     Target device. Default MYRIAD\n");
    //fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}


int main(int argc, char *argv[])
{
    struct stat st;
    lib_detect_info libdetect;
    const char *progname = "";
    const char *alt_detect_lib_path = NULL;
    const char *config_file = NULL;
    const char *input_jpeg_file = NULL;
    int opt;
    int ret = 0;

    progname = argv[0];
    memset(&libdetect, 0, sizeof(libdetect));

    while (((opt = getopt(argc, argv, "l:c:i:h")) != -1))
    {
        switch (opt)
        {
            case 'l':
                alt_detect_lib_path = optarg;
                break;
            case 'c':
                config_file = optarg;
                break;
            case 'i':
                input_jpeg_file = optarg;
                break;
            case 'h': // fall through
            default:
                syntax(progname);
                break;
        }
    }

    // too many arguments given
    if (optind < argc) {
        fprintf(stderr, "Error: Too many arguments given\n");
        exit(EXIT_FAILURE);
    }

    if (!alt_detect_lib_path)
    {
        fprintf(stderr, "Error: detection library not defined\n");
        exit(EXIT_FAILURE);
    }

    if (!input_jpeg_file)
    {
        fprintf(stderr, "Error: input JPEG file not specified\n");
        exit(EXIT_FAILURE);
    }
    if (stat(input_jpeg_file, &st) != 0)
    {
        fprintf(stderr, "Error: %s does not exist\n", input_jpeg_file);
        exit(EXIT_FAILURE);
    }

    if (!config_file)
    {
        fprintf(stderr, "Error: detection library config file not specified\n");
        exit(EXIT_FAILURE);
    }
    if (stat(config_file, &st) != 0)
    {
        fprintf(stderr, "Error: %s does not exist\n", config_file);
        exit(EXIT_FAILURE);
    }


    //unsigned char *output_buffer = NULL;
    //unsigned char *yuv_image = NULL;
    //int output_width = 640;
    //int output_height = 480;
    //yuv_image = malloc(output_width*output_height+output_width*output_height/2);
    //int fd = open(input_jpeg_file, O_RDONLY);
    //read(fd,yuv_image,output_width*output_height+output_width*output_height/2);
    //close(fd);

    alt_detect_result_t alt_detect_result;
    memset(&alt_detect_result, 0, sizeof(alt_detect_result));

    unsigned char *output_buffer = NULL;
    int output_width = 0;
    int output_height = 0;
    int output_num_channels = 0;
    unsigned char *yuv_image = NULL;
    if (!read_JPEG_file(input_jpeg_file, &output_buffer, &output_width, &output_height, &output_num_channels))
    {
        fprintf(stderr, "Error: failed to decompress JPEG file %s\n", input_jpeg_file);
        exit(EXIT_FAILURE);
    }
    printf("output_width: %d\n", output_width);
    printf("output_height: %d\n", output_height);
    printf("output_num_channels: %d\n", output_num_channels);
    output_height &= (~3);
    printf("adjusted output_height: %d\n", output_height);
    yuv_image = rgb2yuv(output_buffer, output_width, output_height, output_num_channels);
    if (!yuv_image)
    {
        ret = -1;
        goto clean_up;
    }

    if (lib_detect_load(&libdetect, alt_detect_lib_path))
    {
        ret = -1;
        goto clean_up;
    }

    printf("Loaded detection library\n");
    if (libdetect.alt_detect_init(config_file))
    {
        fprintf(stderr, "Error: failed to initialize detection library\n");
        ret = -1;
        goto clean_up;
    }
    printf("process image\n");
    libdetect.alt_detect_process(yuv_image, output_width, output_height);

    printf("wait for result ...\n");
    while (!libdetect.alt_detect_result_ready()) {
        sleep(1);
        printf("wait for result ...\n");
    }
    printf("result ready\n");

    printf("get result\n");
    libdetect.alt_detect_get_result(&alt_detect_result);
    printf("got result\n");
    // TODO: do something with the result
    libdetect.alt_detect_free_result(&alt_detect_result);

    libdetect.alt_detect_uninit();

clean_up:
    lib_detect_unload(&libdetect);
    if (output_buffer)
        free(output_buffer);
    if (yuv_image)
        free(yuv_image);
    exit(ret);
}
