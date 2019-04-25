/*

	My Image Manipulation Tools for Digital Humanities
	Testbed of Background Isolation based on Sauvola's Algorithm

	isolate-bg.cpp

	Copyright (C) 2019 Tsukasa OI.

	------------------------------------------------------------------------

	Permission to use, copy, modify, and/or distribute this software for
	any purpose with or without fee is hereby granted, provided that the
	above copyright notice and this permission notice appear in all copies.

	THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
	WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
	MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
	ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
	WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
	ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
	OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

*/

/*

	THIS PROGRAM IS EXPERIMENTAL!

	This program implements following algorithm(s) for internal processing:
	Oliveira et al. (2001) "Fast Digital Image Inpainting"

	It depends on:
	- C++11-compatible compiler
	- GNU-compatible getopt_long function.
	- OpenCV 3.x

*/

#define SOFTWARE_VERSION    "0.0.14"
#define SOFTWARE_COPYRIGHT  "Copyright (C) 2019 Tsukasa OI."

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <getopt.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "microlib/argparse.hpp"

using namespace std;
using namespace cv;



#if 1
// 16843009^2 * 255^2 < 2^64
static const long integralWindowSizeLimit = 16843009;
typedef uint_least64_t intimage_type;
#else
// 257^2 * 255^2 < 2^32
static const long integralWindowSizeLimit = 257;
typedef uint_least32_t intimage_type;
#endif

enum ProgramMode
{
	OUT_NORMALIZED_IMAGE,
	OUT_BACKGROUND,
};

enum InpaintInitMode
{
	ISOBG_INPAINT_INIT_MEAN,
	ISOBG_INPAINT_INIT_NEIGHBOR_L1,
};


static const int    defaultIntegralWindowSize = 60;
static const double defaultKParam = 0.4;
static_assert(defaultIntegralWindowSize <= integralWindowSizeLimit, "defaultIntegralWindowSize must not exceed integralWindowSizeLimit.");

static const int defaultInpaintIterations = 16;

static const double defaultMaskDenoiseDistance1 = 1.0;
static const double defaultMaskDenoiseDistance2 = 5.0;

static const int    defaultBackgroundBlur  = 9;
static const double defaultBackgroundAlpha = 0.9;


static ProgramMode programMode = OUT_NORMALIZED_IMAGE;
static const char* filename_in;
static const char* filename_out;
static bool inputAsGrayscale = false;
static bool adjustBrightness = false;

static int    integralWindowSize = defaultIntegralWindowSize;
static double kParam   = defaultKParam;
static double rScale   = 1.0;

static InpaintInitMode inpaintInitMode   = ISOBG_INPAINT_INIT_NEIGHBOR_L1;
static int             inpaintIterations = defaultInpaintIterations;

static double maskDenoiseDistance1 = defaultMaskDenoiseDistance1;
static double maskDenoiseDistance2 = defaultMaskDenoiseDistance2;

static int    backgroundBlur  = defaultBackgroundBlur;
static double backgroundAlpha = defaultBackgroundAlpha;



static void usage(int argc, char** argv, int ret = 1)
{
	fprintf(stderr,
		"usage: %s \\\n"
		"      [-g] [-w WINDOW_SIZE] [-k K] [-r RSCALE] \\\n"
		"      [-I IIMODE] [-i ITER] [-j DIST1] [-J DIST2] \\\n"
		"      [-A BLUR] [-a ALPHA] \\\n"
		"      [-B] [-G] IN OUT\n"
		"\n"
		"Options:\n"
		"   -h | --help      show this help\n"
		"   -v | --version   show version information\n"
		"   -g               input as grayscale image\n"
		"   -w WINDOW_SIZE   set window size          [%d]\n"
		"   -k K             set K parameter for Sauvola's algorithm     [%f]\n"
		"   -r RSCALE        set scale of R parameter [1.0]\n"
		"                    (1.0 for maximum standard deviation possible)\n"
		"   -I IIMODE        set background inpaint initialization mode\n"
		"                    (mean: mean for whole unmasked image, neighbor: neighbor by L1)\n"
		"   -i ITER          set inpaint iterations   [%d]\n"
		"   -j DIST1         set mask denoise distance (mask shrinking)  [%f]\n"
		"   -J DIST2         set mask denoise distance (mask growing)    [%f]\n"
		"   -A BLUR          set blur size of resulting background       [%d]\n"
		"   -a ALPHA         set normal intensity of background          [%f]\n"
		"   -B               write background image instead of normalized image\n"
		"   -G               adjust brightness of output image\n",
		argv[0],
		defaultIntegralWindowSize, defaultKParam, defaultInpaintIterations,
		defaultMaskDenoiseDistance1, defaultMaskDenoiseDistance2,
		defaultBackgroundBlur, defaultBackgroundAlpha);
	exit(ret);
}

static void argparse(int argc, char** argv)
{
	unordered_map<string, InpaintInitMode> iimodes = {
		{ "mean", ISOBG_INPAINT_INIT_MEAN },
		{ "nearest",     ISOBG_INPAINT_INIT_NEIGHBOR_L1 },
		{ "neighbor",    ISOBG_INPAINT_INIT_NEIGHBOR_L1 },
		{ "neighbor-L1", ISOBG_INPAINT_INIT_NEIGHBOR_L1 },
		{ "default",     ISOBG_INPAINT_INIT_NEIGHBOR_L1 },
	};
	const struct option longopts[] = {
		{ "help",               no_argument, 0, 'h' },
		{ "version",            no_argument, 0, 'v' },
		{ "input-as-grayscale", no_argument, 0, 'g' },
		{ "window-size",        required_argument, 0, 'w' },
		{ "k-param",            required_argument, 0, 'k' },
		{ "r-scale",            required_argument, 0, 'r' },
		{ "inpaint-initmode",   required_argument, 0, 'I' },
		{ "iteration",          required_argument, 0, 'i' },
		{ "mask-denoise-dist1", required_argument, 0, 'j' },
		{ "mask-denoise-dist2", required_argument, 0, 'J' },
		{ "background-blur",    required_argument, 0, 'A' },
		{ "background-alpha",   required_argument, 0, 'a' },
		{},
	};
	int opt, longindex;
	try
	{
		opterr = 0;
		while ((opt = getopt_long(argc, argv, ":hvgw:k:r:I:i:j:J:A:a:BG0123456789", longopts, &longindex)) != -1)
		{
			switch (opt)
			{
				case 'h':
					usage(argc, argv, 0);
					break;
				case 'v':
					fprintf(stderr, "isolate-bg version %s\n%s\n", SOFTWARE_VERSION, SOFTWARE_COPYRIGHT);
					exit(0);
					break;
				case 'g':
					inputAsGrayscale = true;
					break;
				case 'w':
					integralWindowSize = argparse_int("-w", optarg);
					if (integralWindowSize < 1)
						throw argparse_error("-w", "window size is too small.");
					if (integralWindowSize > integralWindowSizeLimit)
						throw argparse_error("-w", "window size is too large.");
					break;
				case 'k':
					kParam = argparse_double("-k", optarg, true);
					if (kParam < 0)
						throw argparse_error("-k", "k parameter is too small.");
					break;
				case 'r':
					rScale = argparse_double("-r", optarg, true);
					if (rScale <= 0)
						throw argparse_error("-r", "R scale must be positive.");
					break;
				case 'I':
				{
					auto p = iimodes.find(optarg);
					if (p == iimodes.end())
						throw argparse_error("-I", "unknown value.");
					inpaintInitMode = p->second;
				}; break;
				case 'i':
					inpaintIterations = argparse_int("-i", optarg);
					if (inpaintIterations < 0)
						throw argparse_error("-i", "inpaint interation count must not be negative.");
					break;
				case 'j':
					maskDenoiseDistance1 = argparse_double("-j", optarg);
					if (maskDenoiseDistance1 < 0)
						throw argparse_error("-j", "denoise distance must not be negative.");
					break;
				case 'J':
					maskDenoiseDistance2 = argparse_double("-J", optarg);
					if (maskDenoiseDistance2 < 0)
						throw argparse_error("-J", "denoise distance must not be negative.");
					break;
				case 'A':
					backgroundBlur = argparse_int("-A", optarg);
					if (backgroundBlur < 1)
						throw argparse_error("-A", "background blur size must be positive.");
					if ((backgroundBlur % 2) != 1)
						throw argparse_error("-A", "background blur size must be an odd integer.");
					break;
				case 'a':
					backgroundAlpha = argparse_double("-a", optarg);
					if (backgroundAlpha < 0 || backgroundAlpha > 1)
						throw argparse_error("-a", "background alpha must be in between 0 and 1.");
					break;
				case 'B':
					programMode = OUT_BACKGROUND;
					break;
				case 'G':
					adjustBrightness = true;
					break;
				// Undocumented presets for testing
				case '1':
					backgroundBlur = 1;
					backgroundAlpha = 1.0;
					break;
				case '0':
				case '2':
				case '3':
				case '4':
				case '5':
				case '6':
				case '7':
				case '8':
				case '9':
					// Not implemented: modify it for your experiments
					break;
				case ':':
					throw argparse_error(argv[0], "insufficient argument.");
					break;
				default: // '?'
					throw argparse_error(argv[0], "invalid option.");
					break;
			}
		}
		if (argc - optind != 2)
			usage(argc, argv, 1);
		filename_in  = argv[optind++];
		filename_out = argv[optind];
	}
	catch (const argparse_error& err)
	{
		fprintf(stderr, "%s: %s\n", err.target.c_str(), err.what_arg.c_str());
		exit(1);
	}
}



static void maskInvert(Mat& img)
{
	int w = img.cols;
	int h = img.rows;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			img.at<unsigned char>(y, x) = 255 - img.at<unsigned char>(y, x);
}

static void maskInset(Mat& img, double width)
{
	Mat tmp;
	int w = img.cols;
	int h = img.rows;
	distanceTransform(img, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			img.at<unsigned char>(y, x) = tmp.at<float>(y, x) <= width ? 0 : 255;
}

static bool binarizeUsingSauvola(Mat& dst, const Mat& src, int integralWindowSize, double kParam, double rScale)
{
	int w = src.cols;
	int h = src.rows;
	double rParam = rScale * (255.0 * 0.5);
	int win_n = integralWindowSize / 2;
	int win_p = integralWindowSize / 2;
	if ((integralWindowSize % 2) != 0)
		++win_n;
	if (
		w == 0 || h == 0 ||
		numeric_limits<int>::max() - integralWindowSize < w ||
		numeric_limits<int>::max() - integralWindowSize < h ||
		numeric_limits<int>::max() / (w + integralWindowSize) < h + integralWindowSize
	)
	{
		return false;
	}
	int pw = w + integralWindowSize;
	int ph = h + integralWindowSize;
	double invsqWindow = 1.0 / integralWindowSize / integralWindowSize;
	dst = Mat(h, w, CV_8U);
	Mat padimg;
	copyMakeBorder(src, padimg, win_n, win_p, win_n, win_p, BORDER_REPLICATE);
	// Make two integral images
	intimage_type* buffer1 = new intimage_type[pw * ph];
	intimage_type* buffer2 = new intimage_type[pw * ph];
	{
		intimage_type accum1 = 0;
		intimage_type accum2 = 0;
		for (int x = 0; x < pw; x++)
		{
			intimage_type value = padimg.at<unsigned char>(0, x);
			accum1 += value;
			accum2 += value * value;
			buffer1[x] = accum1;
			buffer2[x] = accum2;
		}
	}
	for (int y = 1; y < ph; y++)
	{
		intimage_type accum1 = 0;
		intimage_type accum2 = 0;
		for (int x = 0; x < pw; x++)
		{
			intimage_type value = padimg.at<unsigned char>(y, x);
			accum1 += value;
			accum2 += value * value;
			buffer1[pw * y + x] = accum1 + buffer1[pw * (y - 1) + x];
			buffer2[pw * y + x] = accum2 + buffer2[pw * (y - 1) + x];
		}
	}
	// Fast Sauvola's algorithm
	for (int y = 0; y < h; y++)
	{
		int Y0 = pw * y;
		int Y1 = pw * (y + integralWindowSize);
		for (int x = 0; x < w; x++)
		{
			intimage_type total1 =
				buffer1[Y1 + (x + integralWindowSize)] -
				buffer1[Y1 + x] +
				buffer1[Y0 + x] -
				buffer1[Y0 + (x + integralWindowSize)];
			intimage_type total2 =
				buffer2[Y1 + (x + integralWindowSize)] -
				buffer2[Y1 + x] +
				buffer2[Y0 + x] -
				buffer2[Y0 + (x + integralWindowSize)];
			double mean   = total1 * invsqWindow;
			double stddev = sqrt(total2 * invsqWindow - mean * mean);
			int threshold = mean * (1 + kParam * (stddev / rParam - 1));
			dst.at<unsigned char>(y, x) = padimg.at<unsigned char>(y + win_n, x + win_n) > threshold ? 255 : 0;
		}
	}
	// Finalization
	delete[] buffer1;
	delete[] buffer2;
	return true;
}

static bool fastInpaint(const Mat& src, const Mat& mask, Mat& dst, InpaintInitMode initMode, int iterations)
{
	static const float a = 0.073235f;
	static const float b = 0.176765f;
	static Mat kernel = (Mat_<float>(3, 3) << a, b, a, b, 0.0f, b, a, b, a);
	Mat tmp, tmp2;
	dst = Mat(src.clone());
	int w = src.cols;
	int h = src.rows;
	// NOTICE: assume no arithmetic overflow occurs while initialization
	switch (initMode)
	{
		case ISOBG_INPAINT_INIT_MEAN:
		{
			uint_least64_t totalR = 0; // also for single channel
			uint_least64_t totalG = 0;
			uint_least64_t totalB = 0;
			uint_least64_t pixels = 0;
			if (src.channels() == 3)
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						if (!mask.at<unsigned char>(y, x))
						{
							Vec3b value = src.at<Vec3b>(y, x);
							totalB += value[0];
							totalG += value[1];
							totalR += value[2];
							++pixels;
						}
					}
				}
			}
			else
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						if (!mask.at<unsigned char>(y, x))
						{
							totalR += src.at<unsigned char>(y, x);
							++pixels;
						}
					}
				}
			}
			if (!pixels)
				return false;
			// Fill with mean
			unsigned char meanR = totalR / pixels;
			unsigned char meanG = totalG / pixels;
			unsigned char meanB = totalB / pixels;
			if (src.channels() == 3)
			{
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						if (mask.at<unsigned char>(y, x))
							dst.at<Vec3b>(y, x) = Vec3b(meanB, meanG, meanR);
			}
			else
			{
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						if (mask.at<unsigned char>(y, x))
							dst.at<unsigned char>(y, x) = meanR;
			}
		}; break;
		case ISOBG_INPAINT_INIT_NEIGHBOR_L1:
		{
			// Check whether unmasked pixel exists
			bool pixelExists = false;
			for (int y = 0; y < h && !pixelExists; y++)
				for (int x = 0; x < w && !pixelExists; x++)
					if (!mask.at<unsigned char>(y, x))
						pixelExists = true;
			if (!pixelExists)
				return false;
			// Fill with neighbor
			#define SUBCODE(T) \
				for (int y = 0; y < h; y++) \
				{ \
					for (int x = 0; x < w; x++) \
					{ \
						if (mask.at<unsigned char>(y, x)) \
						{ \
							bool ok = false; \
							for (int d = 1; !ok; d++) \
							{ \
								for (int k = 0; k < d && !ok; k++) \
								{ \
									int X, Y; \
									X = x + k; Y = y - d + k; \
									if (X >= 0 && X < w && Y >= 0 && Y < h && !mask.at<unsigned char>(Y, X)) \
									{ \
										dst.at<T>(y, x) = dst.at<T>(Y, X); ok = true; break; \
									} \
									X = x + d - k; Y = y + k; \
									if (X >= 0 && X < w && Y >= 0 && Y < h && !mask.at<unsigned char>(Y, X)) \
									{ \
										dst.at<T>(y, x) = dst.at<T>(Y, X); ok = true; break;\
									} \
									X = x - k; Y = y + d - k; \
									if (X >= 0 && X < w && Y >= 0 && Y < h && !mask.at<unsigned char>(Y, X)) \
									{ \
										dst.at<T>(y, x) = dst.at<T>(Y, X); ok = true; break; \
									} \
									X = x - d + k; Y = y - k; \
									if (X >= 0 && X < w && Y >= 0 && Y < h && !mask.at<unsigned char>(Y, X)) \
									{ \
										dst.at<T>(y, x) = dst.at<T>(Y, X); ok = true; break; \
									} \
								} \
							} \
						} \
					} \
				}
			if (src.channels() == 3)
			{
				SUBCODE(Vec3b)
			}
			else
			{
				SUBCODE(unsigned char)
			}
			#undef SUBCODE
		}; break;
	}
	dst.convertTo(dst, src.channels() == 3 ? CV_32FC3 : CV_32F);
	for (int i = 0; i < iterations; i++)
	{
		filter2D(dst, tmp, -1, kernel, Point(1, 1), 0, BORDER_REPLICATE);
		if (src.channels() == 3)
		{
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					if (mask.at<unsigned char>(y, x))
						dst.at<Vec3f>(y, x) = tmp.at<Vec3f>(y, x);
		}
		else
		{
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
					if (mask.at<unsigned char>(y, x))
						dst.at<float>(y, x) = tmp.at<float>(y, x);
		}
	}
	dst.convertTo(dst, src.channels() == 3 ? CV_8UC3 : CV_8U);
	return true;
}



int main(int argc, char** argv)
{
	argparse(argc, argv);
	Mat img = imread(filename_in, IMREAD_ANYCOLOR);
	if (!img.data)
	{
		fprintf(stderr, "%s: image could not be loaded.\n", filename_in);
		return 1;
	}
	if (inputAsGrayscale)
		cvtColor(img, img, CV_BGR2GRAY);
	int w = img.cols;
	int h = img.rows;
	// Background inpainting and isolation
	Mat bg;
	{
		Mat tmp, tmp2;
		if (img.channels() == 3)
			cvtColor(img, tmp2, CV_BGR2GRAY);
		else
			tmp2 = img;
		binarizeUsingSauvola(tmp, tmp2, integralWindowSize, kParam, rScale);
		maskInvert(tmp);
		maskInset(tmp, maskDenoiseDistance1);
		maskInvert(tmp);
		maskInset(tmp, maskDenoiseDistance2);
		maskInvert(tmp);
		fastInpaint(img, tmp, bg, inpaintInitMode, inpaintIterations);
	}
	if (backgroundBlur != 1)
		GaussianBlur(bg, bg, Size(backgroundBlur, backgroundBlur), 0.0, 0.0, BORDER_REPLICATE);
	// Output
	switch (programMode)
	{
		case OUT_NORMALIZED_IMAGE:
			// Normalize original image by background
			if (img.channels () == 3)
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						Vec3b cI = img.at<Vec3b>(y, x);
						Vec3b cB =  bg.at<Vec3b>(y, x);
						Vec3b cR = Vec3b(
							min(max(backgroundAlpha * cI[0] / cB[0], 0.0), 1.0) * 255.0,
							min(max(backgroundAlpha * cI[1] / cB[1], 0.0), 1.0) * 255.0,
							min(max(backgroundAlpha * cI[2] / cB[2], 0.0), 1.0) * 255.0
						);
						img.at<Vec3b>(y, x) = cR;
					}
				}
			}
			else
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						unsigned char vI = img.at<unsigned char>(y, x);
						unsigned char vB =  bg.at<unsigned char>(y, x);
						unsigned char vR = min(max(backgroundAlpha * vI / vB, 0.0), 1.0) * 255.0;
						img.at<unsigned char>(y, x) = vR;
					}
				}
			}
			break;
		case OUT_BACKGROUND:
			img = bg;
			break;
	}
	if (adjustBrightness)
	{
		double Emin, Emax;
		{
			Mat tmp;
			if (img.channels() == 3)
				cvtColor(img, tmp, CV_BGR2GRAY);
			else
				tmp = img;
			unsigned char* p = tmp.data;
			size_t l = tmp.total();
			Emin = *(min_element(p, p + l));
			Emax = *(max_element(p, p + l));
		}
		if (Emin < Emax)
		{
			double Escale = 255.0 / (Emax - Emin);
			if (img.channels() == 3)
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						Vec3b cI = img.at<Vec3b>(y, x);
						Vec3b cR = Vec3b(
							min(max((cI[0] - Emin) * Escale, 0.0), 255.0),
							min(max((cI[1] - Emin) * Escale, 0.0), 255.0),
							min(max((cI[2] - Emin) * Escale, 0.0), 255.0)
						);
						img.at<Vec3b>(y, x) = cR;
					}
				}
			}
			else
			{
				for (int y = 0; y < h; y++)
				{
					for (int x = 0; x < w; x++)
					{
						unsigned char vI = img.at<unsigned char>(y, x);
						unsigned char vR = (vI - Emin) * Escale;
						img.at<unsigned char>(y, x) = vR;
					}
				}
			}
		}
	}
	imwrite(filename_out, img);
	return 0;
}
