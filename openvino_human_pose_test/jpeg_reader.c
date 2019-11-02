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

#include "jpeglib.h"
#include <setjmp.h> // optional error recovery mechanism

#include "jpeg_reader.h"


#define CLIP(X) ( (X) > 255 ? 255 : (X) < 0 ? 0 : X)

//// RGB -> YUV
//#define RGB2Y(R, G, B) CLIP(( (  66 * (R) + 129 * (G) +  25 * (B) + 128) >> 8) +  16)
//#define RGB2U(R, G, B) CLIP(( ( -38 * (R) -  74 * (G) + 112 * (B) + 128) >> 8) + 128)
//#define RGB2V(R, G, B) CLIP(( ( 112 * (R) -  94 * (G) -  18 * (B) + 128) >> 8) + 128)
//
//// YUV -> RGB
//#define C(Y) ( (Y) - 16  )
//#define D(U) ( (U) - 128 )
//#define E(V) ( (V) - 128 )
//
//#define YUV2R(Y, U, V) CLIP(( 298 * C(Y)              + 409 * E(V) + 128) >> 8)
//#define YUV2G(Y, U, V) CLIP(( 298 * C(Y) - 100 * D(U) - 208 * E(V) + 128) >> 8)
//#define YUV2B(Y, U, V) CLIP(( 298 * C(Y) + 516 * D(U)              + 128) >> 8)
//
// RGB -> YCbCr
#define CRGB2Y(R, G, B) CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16)
#define CRGB2Cb(R, G, B) CLIP((36962 * (B - CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16) ) >> 16) + 128)
#define CRGB2Cr(R, G, B) CLIP((46727 * (R - CLIP((19595 * R + 38470 * G + 7471 * B ) >> 16) ) >> 16) + 128)

//// YCbCr -> RGB
//#define CYCbCr2R(Y, Cb, Cr) CLIP( Y + ( 91881 * Cr >> 16 ) - 179 )
//#define CYCbCr2G(Y, Cb, Cr) CLIP( Y - (( 22544 * Cb + 46793 * Cr ) >> 16) + 135)
//#define CYCbCr2B(Y, Cb, Cr) CLIP( Y + (116129 * Cb >> 16 ) - 226 )



/******************** JPEG DECOMPRESSION SAMPLE INTERFACE *******************/

/* This half of the example shows how to read data from the JPEG decompressor.
 * It's a bit more refined than the above, in that we show:
 *   (a) how to modify the JPEG library's standard error-reporting behavior;
 *   (b) how to allocate workspace using the library's memory manager.
 *
 * Just to make this example a little different from the first one, we'll
 * assume that we do not intend to put the whole image into an in-memory
 * buffer, but to send it line-by-line someplace else.  We need a one-
 * scanline-high JSAMPLE array as a work buffer, and we will let the JPEG
 * memory manager allocate it for us.  This approach is actually quite useful
 * because we don't need to remember to deallocate the buffer separately: it
 * will go away automatically when the JPEG object is cleaned up.
 */


/*
 * ERROR HANDLING:
 *
 * The JPEG library's standard error handler (jerror.c) is divided into
 * several "methods" which you can override individually.  This lets you
 * adjust the behavior without duplicating a lot of code, which you might
 * have to update with each future release.
 *
 * Our example here shows how to override the "error_exit" method so that
 * control is returned to the library's caller when a fatal error occurs,
 * rather than calling exit() as the standard error_exit method does.
 *
 * We use C's setjmp/longjmp facility to return control.  This means that the
 * routine which calls the JPEG library must first execute a setjmp() call to
 * establish the return point.  We want the replacement error_exit to do a
 * longjmp().  But we need to make the setjmp buffer accessible to the
 * error_exit routine.  To do this, we make a private extension of the
 * standard JPEG error handler object.  (If we were using C++, we'd say we
 * were making a subclass of the regular error handler.)
 *
 * Here's the extended error handler struct:
 */

struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */

  jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct my_error_mgr * my_error_ptr;

/*
 * Here's the routine that will replace the standard error_exit method:
 */

METHODDEF(void)
my_error_exit (j_common_ptr cinfo)
{
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr) cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  (*cinfo->err->output_message) (cinfo);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}


/*
 * Sample routine for JPEG decompression.  We assume that the source file name
 * is passed in.  We want to return 1 on success, 0 on error.
 */


int read_JPEG_file (const char * filename,
                    unsigned char **output_buffer,
                    int *output_width,
                    int *output_height,
                    int *output_num_channels)
{
  *output_buffer = NULL;
  *output_width = 0;
  *output_height = 0;
  *output_num_channels = 0;

  /* This struct contains the JPEG decompression parameters and pointers to
   * working space (which is allocated as needed by the JPEG library).
   */
  struct jpeg_decompress_struct cinfo;
  /* We use our private extension JPEG error handler.
   * Note that this struct must live as long as the main JPEG parameter
   * struct, to avoid dangling-pointer problems.
   */
  struct my_error_mgr jerr;
  /* More stuff */
  FILE * infile;		/* source file */
  JSAMPARRAY buffer;		/* Output row buffer */
  int row_stride;		/* physical row width in output buffer */

  /* In this example we want to open the input file before doing anything else,
   * so that the setjmp() error recovery below can assume the file is open.
   * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
   * requires it in order to read binary files.
   */

  if ((infile = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return 0;
  }

  /* Step 1: allocate and initialize JPEG decompression object */

  /* We set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object, close the input file, and return.
     */
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
  }
  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */

  jpeg_stdio_src(&cinfo, infile);

  /* Step 3: read file parameters with jpeg_read_header() */

  (void) jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.txt for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  /* Step 5: Start decompressor */

  (void) jpeg_start_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* We may need to do some setup of our own at this point before reading
   * the data.  After jpeg_start_decompress() we have the correct scaled
   * output image dimensions available, as well as the output colormap
   * if we asked for color quantization.
   * In this example, we need to make an output work buffer of the right size.
   */
  /* JSAMPLEs per row in output buffer */
  row_stride = cinfo.output_width * cinfo.output_components;
  /* Make a one-row-high sample array that will go away when done with image */
  buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */

  int w = cinfo.output_width;
  int h = cinfo.output_height;
  int numChannels = cinfo.num_components; // 3 = RGB, 4 = RGBA
  unsigned long dataSize = w * h * numChannels;

  // read RGB(A) scanlines one at a time into jdata[]
  unsigned char *data = (unsigned char *)malloc( dataSize );
  unsigned char* rowptr;

  if (data == NULL) {
    fprintf(stderr, "Failed to allocate memory for decompressed JPEG image\n");
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
  }
  /* Here we use the library's state variable cinfo.output_scanline as the
   * loop counter, so that we don't have to keep track ourselves.
   */
  while (cinfo.output_scanline < cinfo.output_height) {
    rowptr = data + cinfo.output_scanline * w * numChannels;
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    /* Assume put_scanline_someplace wants a pointer and sample count. */
    //put_scanline_someplace(buffer[0], row_stride);
    memcpy(rowptr, buffer[0], w * numChannels);
  }
  *output_buffer = data;
  *output_width = w;
  *output_height = h;
  *output_num_channels = numChannels;

  /* Step 7: Finish decompression */

  (void) jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  /* After finish_decompress, we can close the input file.
   * Here we postpone it until after no more JPEG errors are possible,
   * so as to simplify the setjmp error logic above.  (Actually, I don't
   * think that jpeg_destroy can do an error exit, but why assume anything...)
   */
  fclose(infile);

  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */
  if (jerr.pub.num_warnings)
    fprintf(stderr, "JPEG decompressed with %ld warnings !\n", jerr.pub.num_warnings);

  /* And we're done! */
  return 1;
}


unsigned char *rgb2yuv(unsigned char *image, int width, int height, int numChannels)
{
    //int output_size = width * height + width * height / 2;
    //unsigned char *output = malloc(output_size);
    //if (output == NULL) {
    //    fprintf(stderr, "Error: failed to allocate memory for RGB to YUV conversion\n");
    //    return NULL;
    //}
    //int yuv444_output_size = width * height * 3;
    //unsigned char *yuv444_output = malloc(yuv444_output_size);
    //if (yuv444_output == NULL) {
    //    fprintf(stderr, "Error: failed to allocate memory for RGB to YUV conversion\n");
    //    free(output);
    //    return NULL;
    //}
    //
    //for (int j = 0; j < height; j++) {
    //    for (int i = 0; i < width; i++) {
    //        int r = image[(i + j * width) * numChannels] ;
    //        int g = image[(i + j * width) * numChannels + 1] ;
    //        int b = image[(i + j * width) * numChannels + 2] ;
    //        yuv444_output[(i + j * width) * 3] = CRGB2Y(r,g,b);
    //        yuv444_output[(i + j * width) * 3 + 1] = CRGB2Cb(r,g,b);
    //        yuv444_output[(i + j * width) * 3 + 2] = CRGB2Cr(r,g,b);
    //    }
    //}
    //
    //// YUV444 to YUV420 conversion
    //int sum;
    //int si0, si1, sj0, sj1;
    //
    //for (int j = 0; j < height; j++) {
    //    for (int i = 0; i < width; i++) {
    //        output[(i + j * width) * 3] = yuv444_output[(i + j * width) * 3] ;
    //    }
    //}
    //
    //for (int j = 0; j < height; j+=2) {
    //    sj0 = j ;
    //    sj1 = (j + 1 < height) ? j + 1 : j ;
    //
    //    for (int i = 0; i < width; i+=2) {
    //        si0 = i ;
    //        si1 = (i + 1 < width) ? i + 1 : i ;
    //
    //        sum =  yuv444_output[(si0 + sj0 * width) * 3 + 1] ;
    //        sum += yuv444_output[(si1 + sj0 * width) * 3 + 1] ;
    //        sum += yuv444_output[(si0 + sj1 * width) * 3 + 1] ;
    //        sum += yuv444_output[(si1 + sj1 * width) * 3 + 1] ;
    //        sum = CLIP(sum / 4) ;
    //
    //        output[(si0 + sj0 * width) * 3 + 1] = sum ;
    //        output[(si1 + sj0 * width) * 3 + 1] = sum ;
    //        output[(si0 + sj1 * width) * 3 + 1] = sum ;
    //        output[(si1 + sj1 * width) * 3 + 1] = sum ;
    //
    //        sum =  yuv444_output[(si0 + sj0 * width) * 3 + 2] ;
    //        sum += yuv444_output[(si1 + sj0 * width) * 3 + 2] ;
    //        sum += yuv444_output[(si0 + sj1 * width) * 3 + 2] ;
    //        sum += yuv444_output[(si1 + sj1 * width) * 3 + 2] ;
    //        sum = CLIP(sum / 4) ;
    //
    //        output[(si0 + sj0 * width) * 3 + 2] = sum ;
    //        output[(si1 + sj0 * width) * 3 + 2] = sum ;
    //        output[(si0 + sj1 * width) * 3 + 2] = sum ;
    //        output[(si1 + sj1 * width) * 3 + 2] = sum ;
    //    }
    //}
    //
    //free(yuv444_output);


//#if 0
    int output_size = width * height + width * height / 2;
    unsigned char *output = malloc(output_size);
    if (output == NULL) {
        fprintf(stderr, "Error: failed to allocate memory for RGB to YUV conversion\n");
        return NULL;
    }

    int image_size = width * height;
    int upos = image_size;
    int vpos = upos + (upos >> 2);

    for (int y = 0; y < height; y++ ) {
        for (int x = 0; x < width; x++) {
            int rgbindex = numChannels * (y*width+x);
            int r = image[rgbindex];
            int g = image[rgbindex + 1];
            int b = image[rgbindex + 2];
            output[y*width+x] = CRGB2Y(r,g,b);
            if ((!(y & 1)) && (!(x & 1))) {
                output[upos++] = CRGB2Cb(r,g,b);
                output[vpos++] = CRGB2Cr(r,g,b);
            }
        }
    }


    //int output_size = width * height + width * height / 2;
    //unsigned char *output = malloc(output_size);
    //if (output == NULL) {
    //    fprintf(stderr, "Error: failed to allocate memory for RGB to YUV conversion\n");
    //    return NULL;
    //}
    //
    //int image_size = width * height;
    //int upos = image_size;
    //int vpos = upos + (upos >> 2);
    ////int i = 0;
    //
    //for(int i = 0; i < image_size; ++i ) {
    //    int r = image[numChannels * i];
    //    int g = image[numChannels * i + 1];
    //    int b = image[numChannels * i + 2];
    //    output[i] = CRGB2Y(r,g,b);
    //    //destination[i] = ( ( 66*r + 129*g + 25*b ) >> 8 ) + 16;
    //    //if (!((i / width) % 2) && !(i % 2)) {
    //    if ((!(i & 1)) && (!((i/width)&1))) {
    //    //if (!(i % 4)) {
    //        output[upos++] = CRGB2Cb(r,g,b);
    //        output[vpos++] = CRGB2Cr(r,g,b);
    //        //destination[upos++] = ( ( -38*r + -74*g + 112*b ) >> 8) + 128;
    //        //destination[vpos++] = ( ( 112*r + -94*g + -18*b ) >> 8) + 128;
    //    }
    //}


    //for( int line = 0; line < height; ++line )
    //{
    //    if( !(line % 2) )
    //    {
    //        for( int x = 0; x < width; x += 2 )
    //        {
    //            int r = image[numChannels * i];
    //            int g = image[numChannels * i + 1];
    //            int b = image[numChannels * i + 2];
    //
    //            //output[i++] = CLIP(((66*r + 129*g + 25*b) >> 8) + 16);
    //            //output[upos++] = CLIP(((-38*r + -74*g + 112*b) >> 8) + 128);
    //            //output[vpos++] = CLIP(((112*r + -94*g + -18*b) >> 8) + 128);
    //            output[i++] = CRGB2Y(r,g,b);
    //            output[upos++] = CRGB2Cb(r,g,b);
    //            output[vpos++] = CRGB2Cr(r,g,b);
    //
    //            r = image[numChannels * i];
    //            g = image[numChannels * i + 1];
    //            b = image[numChannels * i + 2];
    //
    //            //output[i++] = CLIP(((66*r + 129*g + 25*b) >> 8) + 16);
    //            output[i++] = CRGB2Y(r,g,b);
    //        }
    //    }
    //    else
    //    {
    //        for( int x = 0; x < width; x += 1 )
    //        {
    //            int r = image[numChannels * i];
    //            int g = image[numChannels * i + 1];
    //            int b = image[numChannels * i + 2];
    //
    //            //output[i++] = CLIP(((66*r + 129*g + 25*b) >> 8) + 16);
    //            output[i++] = CRGB2Y(r,g,b);
    //        }
    //    }
    //}

//#endif
    return output;
}
