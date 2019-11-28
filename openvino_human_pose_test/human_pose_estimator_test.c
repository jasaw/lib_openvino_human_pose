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

#define MAX(a,b) (a>b?a:b)


#define STATE_NOT_PROCESSED     0
#define STATE_PROCESSING        1
#define STATE_DONE              2


typedef struct
{
    void *handle;
    int (*alt_detect_init)(const char *);
    void (*alt_detect_uninit)();
    int (*alt_detect_process_yuv420)(int, unsigned char *, int, int);
    int (*alt_detect_result_ready)(int);
    int (*alt_detect_get_result)(int, float, alt_detect_result_t *);
    void (*alt_detect_free_result)(alt_detect_result_t *);
    const char *(*alt_detect_err_msg)(void);
    int (*alt_detect_save_yuv420)(unsigned char *, int, int, const char *);
    int (*alt_detect_render_save_yuv420)(unsigned char *, int, int,
                                         alt_detect_result_t *, const char *);

} lib_detect_info;


typedef struct
{
    const char *input_jpeg_file;
    unsigned char *output_buffer;
    unsigned char *yuv_image;
    char *output_file;
    int width;
    int height;
    int id;
    int state;
} image_t;


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
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_process_yuv420), libdetect->handle, "alt_detect_process_yuv420");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_result_ready), libdetect->handle, "alt_detect_result_ready");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_get_result), libdetect->handle, "alt_detect_get_result");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_free_result), libdetect->handle, "alt_detect_free_result");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_err_msg), libdetect->handle, "alt_detect_err_msg");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_save_yuv420), libdetect->handle, "alt_detect_save_yuv420");
    err |= lib_detect_load_sym((void **)(&libdetect->alt_detect_render_save_yuv420), libdetect->handle, "alt_detect_render_save_yuv420");
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


static inline void clip_pos_to_image_size(int *xpos, int *ypos, int width, int height)
{
    if (*xpos < 0)
        *xpos = 0;
    else if (*xpos > width - 1)
        *xpos = width - 1;

    if (*ypos < 0)
        *ypos = 0;
    else if (*ypos > height - 1)
        *ypos = height - 1;
}


static void overlay_result_on_image(unsigned char *yuv_image,
                                    int width,
                                    int height,
                                    alt_detect_result_t *alt_detect_result)
{
    for (int i = 0; i < alt_detect_result->num_objs; i++) {
        alt_detect_obj_t *cur_obj = &alt_detect_result->objs[i];
        printf("Object %d score: %f\n", i+1, cur_obj->score);
        for (int j = 0; j < cur_obj->num_lines; j++) {
            alt_detect_line_t *cur_line = &cur_obj->lines[j];

            //printf("\t%2d:(%d, %d) --- %2d:(%d, %d)\n",
            //       cur_line->p[0].id, cur_line->p[0].x, cur_line->p[0].y,
            //       cur_line->p[1].id, cur_line->p[1].x, cur_line->p[1].y);

            // colour
            unsigned char colour_y = 0;
            unsigned char colour_u = 0;
            unsigned char colour_v = 0;
            switch (cur_line->p[1].id) {
                case  0: colour_y = CRGB2Y(  0,  0,255); colour_u = CRGB2Cb(  0,  0,255); colour_v = CRGB2Cr(  0,  0,255); break;
                case  1: colour_y = CRGB2Y(  0, 85,255); colour_u = CRGB2Cb(  0, 85,255); colour_v = CRGB2Cr(  0, 85,255); break;
                case  2: colour_y = CRGB2Y(  0,170,255); colour_u = CRGB2Cb(  0,170,255); colour_v = CRGB2Cr(  0,170,255); break;
                case  3: colour_y = CRGB2Y(  0,255,255); colour_u = CRGB2Cb(  0,255,255); colour_v = CRGB2Cr(  0,255,255); break;
                case  4: colour_y = CRGB2Y(  0,255,170); colour_u = CRGB2Cb(  0,255,170); colour_v = CRGB2Cr(  0,255,170); break;
                case  5: colour_y = CRGB2Y(  0,255, 85); colour_u = CRGB2Cb(  0,255, 85); colour_v = CRGB2Cr(  0,255, 85); break;
                case  6: colour_y = CRGB2Y(  0,255,  0); colour_u = CRGB2Cb(  0,255,  0); colour_v = CRGB2Cr(  0,255,  0); break;
                case  7: colour_y = CRGB2Y( 85,255,  0); colour_u = CRGB2Cb( 85,255,  0); colour_v = CRGB2Cr( 85,255,  0); break;
                case  8: colour_y = CRGB2Y(170,255,  0); colour_u = CRGB2Cb(170,255,  0); colour_v = CRGB2Cr(170,255,  0); break;
                case  9: colour_y = CRGB2Y(255,255,  0); colour_u = CRGB2Cb(255,255,  0); colour_v = CRGB2Cr(255,255,  0); break;
                case 10: colour_y = CRGB2Y(255,170,  0); colour_u = CRGB2Cb(255,170,  0); colour_v = CRGB2Cr(255,170,  0); break;
                case 11: colour_y = CRGB2Y(255, 85,  0); colour_u = CRGB2Cb(255, 85,  0); colour_v = CRGB2Cr(255, 85,  0); break;
                case 12: colour_y = CRGB2Y(255,  0,  0); colour_u = CRGB2Cb(255,  0,  0); colour_v = CRGB2Cr(255,  0,  0); break;
                case 13: colour_y = CRGB2Y(255,  0, 85); colour_u = CRGB2Cb(255,  0, 85); colour_v = CRGB2Cr(255,  0, 85); break;
                case 14: colour_y = CRGB2Y(255,  0,170); colour_u = CRGB2Cb(255,  0,170); colour_v = CRGB2Cr(255,  0,170); break;
                case 15: colour_y = CRGB2Y(255,  0,255); colour_u = CRGB2Cb(255,  0,255); colour_v = CRGB2Cr(255,  0,255); break;
                case 16: colour_y = CRGB2Y(170,  0,255); colour_u = CRGB2Cb(170,  0,255); colour_v = CRGB2Cr(170,  0,255); break;
                case 17: colour_y = CRGB2Y( 85,  0,255); colour_u = CRGB2Cb( 85,  0,255); colour_v = CRGB2Cr( 85,  0,255); break;
            }

            int stick_width = 3;
            for (int sw = 0; sw < stick_width; sw++) {
                alt_detect_point_t p[2];
                for (int pi = 0; pi < 2; pi++) {
                    p[pi].id = cur_line->p[pi].id;
                    p[pi].x  = cur_line->p[pi].x - (stick_width>>1) + sw;
                    p[pi].y  = cur_line->p[pi].y - (stick_width>>1) + sw;
                }
                long ySize = width * height;
                long uSize = ySize >> 2;
                unsigned char *Y = yuv_image;
                unsigned char *U = Y + ySize;
                unsigned char *V = U + uSize;
                // draw on image
                int x_len = p[1].x - p[0].x;
                int y_len = p[1].y - p[0].y;
                //printf("x_len: %d, y_len: %d\n", x_len, y_len);
                int num_steps = MAX(abs(x_len), abs(y_len));
                double x_step = x_len / ((double)num_steps);
                double y_step = y_len / ((double)num_steps);
                //printf("x_step: %f, y_step: %f\n", x_step, y_step);
                double dx = p[0].x;
                double dy = p[0].y;
                for (int s = 0; s <= num_steps; s++) {
                    int Yx_pos = (int)dx;
                    int Yy_pos = (int)dy;
                    int Yindex = Yy_pos*width+Yx_pos;
                    clip_pos_to_image_size(&Yx_pos, &Yy_pos, width, height);
                    int UVx_pos = Yx_pos >> 1;
                    int UVy_pos = Yy_pos >> 1;
                    int UVindex = UVy_pos*(width>>1)+UVx_pos;
                    //printf("dx: %f, dy: %f\n", dx, dy);
                    //Y[Yindex] = ~Y[Yindex];
                    Y[Yindex]  = colour_y;
                    U[UVindex] = colour_u;
                    V[UVindex] = colour_v;
                    dx += x_step;
                    dy += y_step;
                }
            }
        }
    }
}


static void syntax(const char *progname)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "%s [options] image1.jpg [image2.jpg] ...\n", progname);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, " -l [libpath]          Detection library path\n");
    fprintf(stderr, " -c [configfile]       Detection library configuration file\n");
    fprintf(stderr, " -s [score]            Score threshold\n");
    fprintf(stderr, " -h                    Display this help page\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Detection Library config keys:\n");
    fprintf(stderr, "MODEL_XML              Detection model XML path\n");
    fprintf(stderr, "MODEL_BIN              Detection model BIN path\n");
    fprintf(stderr, "TARGET_DEVICE          Target device. Default to MYRIAD if not set.\n");
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}


int main(int argc, char *argv[])
{
    const char *output_file_prefix = "out_";
    const char *output_file_suffix = ".png";
    float score_threshold = 200;
    struct stat st;
    lib_detect_info libdetect;
    const char *progname = "";
    const char *alt_detect_lib_path = NULL;
    const char *config_file = NULL;
    int opt;
    int ret = 0;
    int i;
    image_t *images = NULL;
    int num_images = 0;
    alt_detect_result_t alt_detect_result;

    progname = argv[0];
    memset(&libdetect, 0, sizeof(libdetect));
    memset(&alt_detect_result, 0, sizeof(alt_detect_result));

    while (((opt = getopt(argc, argv, "l:c:s:h")) != -1))
    {
        switch (opt)
        {
            case 'l':
                alt_detect_lib_path = optarg;
                break;
            case 'c':
                config_file = optarg;
                break;
            case 's':
                score_threshold = atof(optarg);
                break;
            case 'h': // fall through
            default:
                syntax(progname);
                break;
        }
    }

    // too few arguments given
    if (optind >= argc) {
        fprintf(stderr, "Error: Too few arguments given\n");
        exit(EXIT_FAILURE);
    }

    num_images = argc - optind;
    images = malloc(num_images * sizeof(image_t));
    if (!images) {
        fprintf(stderr, "Error: failed to allocate memory for input images\n");
        exit(EXIT_FAILURE);
    }
    memset(images, 0, sizeof(image_t) * num_images);
    for (i = 0; i < num_images; i++) {
        images[i].input_jpeg_file = argv[optind + i];
        images[i].id = num_images-i-1;
        images[i].output_file = malloc(strlen(images[i].input_jpeg_file) +
                                       strlen(output_file_prefix) +
                                       strlen(output_file_suffix) + 1);
        if (!images[i].output_file) {
            fprintf(stderr, "Error: failed to allocate memory for input images\n");
            ret = -1;
            goto clean_up;
        }
        strcpy(images[i].output_file, output_file_prefix);
        strcpy(images[i].output_file + strlen(output_file_prefix), images[i].input_jpeg_file);
        strcpy(images[i].output_file + strlen(output_file_prefix) + strlen(images[i].input_jpeg_file), output_file_suffix);
    }
    for (i = 0; i < num_images; i++) {
        if (stat(images[i].input_jpeg_file, &st) != 0)
        {
            fprintf(stderr, "Error: %s does not exist\n", images[i].input_jpeg_file);
            ret = -1;
            goto clean_up;
        }
    }

    if (!alt_detect_lib_path)
    {
        fprintf(stderr, "Error: detection library not defined\n");
        ret = -1;
        goto clean_up;
    }

    if (!config_file)
    {
        fprintf(stderr, "Error: detection library config file not specified\n");
        ret = -1;
        goto clean_up;
    }
    if (stat(config_file, &st) != 0)
    {
        fprintf(stderr, "Error: %s does not exist\n", config_file);
        ret = -1;
        goto clean_up;
    }


    printf("Score: %f\n", score_threshold);
    printf("\n");


    for (i = 0; i < num_images; i++) {
        int output_num_channels = 0;
        printf("Image %d: %s\n", i, images[i].input_jpeg_file);
        if (!read_JPEG_file(images[i].input_jpeg_file, &images[i].output_buffer,
                            &images[i].width, &images[i].height, &output_num_channels))
        {
            fprintf(stderr, "Error: failed to decompress JPEG file %s\n", images[i].input_jpeg_file);
            ret = -1;
            goto clean_up;
        }

        printf("\tresolution: %d x %d\n", images[i].width, images[i].height);
        int new_height = images[i].height & (~3);
        //int new_width  = images[i].width  & (~3);
        int new_width  = images[i].width;
        if ((new_height != images[i].height) || (new_width != images[i].width)) {
            images[i].height = new_height;
            images[i].width  = new_width;
            printf("\tadjusted resolution to: %d x %d\n", images[i].width, images[i].height);
        }
        printf("\toutput_num_channels: %d\n", output_num_channels);

        images[i].yuv_image = rgb2yuv(images[i].output_buffer, images[i].width,
                                      images[i].height, output_num_channels);
        if (!images[i].yuv_image)
        {
            fprintf(stderr, "Error: failed to convert %s from RGB to YUV\n", images[i].input_jpeg_file);
            ret = -1;
            goto clean_up;
        }
    }



    if (lib_detect_load(&libdetect, alt_detect_lib_path))
    {
        ret = -1;
        goto clean_up;
    }
    printf("Loaded detection library\n");
    if (libdetect.alt_detect_init(config_file))
    {
        const char *errmsg = libdetect.alt_detect_err_msg();
        fprintf(stderr, "Error: %s\n", errmsg);
        ret = -1;
        goto clean_up;
    }

    printf("\n");

    int done = 0;
    while (!done)
    {
        for (i = 0; i < num_images; i++) {
            if (images[i].state == STATE_NOT_PROCESSED) {
                if (libdetect.alt_detect_process_yuv420(images[i].id,
                                                        images[i].yuv_image,
                                                        images[i].width,
                                                        images[i].height) == 0) {
                    printf("process image %s\n", images[i].input_jpeg_file);
                    images[i].state = STATE_PROCESSING;
                }
            }
            if (libdetect.alt_detect_result_ready(images[i].id)) {
                printf("get results %s\n", images[i].input_jpeg_file);
                if (libdetect.alt_detect_get_result(images[i].id,
                                                    score_threshold,
                                                    &alt_detect_result) >= 0) {
                    printf("overlay result on image %s\n", images[i].output_file);
                    overlay_result_on_image(images[i].yuv_image, images[i].width,
                                            images[i].height, &alt_detect_result);
                    libdetect.alt_detect_save_yuv420(images[i].yuv_image,
                                                     images[i].width,
                                                     images[i].height, images[i].output_file);
                    images[i].state = STATE_DONE;
                }
            }
        }

        done = 1;
        for (i = 0; i < num_images; i++) {
            if (images[i].state != STATE_DONE) {
                done = 0;
                break;
            }
        }

        if (!done)
            usleep(100*1000);
    }


clean_up:
    printf("free result\n");
    libdetect.alt_detect_free_result(&alt_detect_result);
    libdetect.alt_detect_uninit();
    lib_detect_unload(&libdetect);
    if (images) {
        for (i = 0; i < num_images; i++) {
            if (images[i].output_buffer)
                free(images[i].output_buffer);
            if (images[i].yuv_image)
                free(images[i].yuv_image);
            if (images[i].output_file)
                free(images[i].output_file);
        }
        free(images);
    }
    exit(ret);
}
