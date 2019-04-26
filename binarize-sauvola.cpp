/*

	My Image Manipulation Tools for Digital Humanities
	Implementation and Testbed of Sauvola's Algorithm

	binarize-sauvola.cpp

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

	This program implements the algorithm described in the paper:
	Shafait et al. (2008) "Efficient Implementation of Local Adaptive Thresholding Techniques Using Integral Images"

	It depends on:
	- C++11-compatible compiler
	- GNU-compatible getopt_long function.
	- OpenCV 3.x

*/

#define SOFTWARE_VERSION    "0.3.2"
#define SOFTWARE_COPYRIGHT  "Copyright (C) 2019 Tsukasa OI."

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <getopt.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "microlib/argparse.hpp"

using namespace std;
using namespace cv;



#if 1
// 16843009^2 * 255^2 < 2^64
static const long windowSizeLimit = 16843009;
typedef uint_least64_t intimage_type;
#else
// 257^2 * 255^2 < 2^32
static const long windowSizeLimit = 257;
typedef uint_least32_t intimage_type;
#endif

enum ProgramMode
{
	OUT_BINARY,
	OUT_THRESHOLD,
	OUT_PIXELINFO,
	OUT_VARIABLE,
	OUT_VARIABLE_MULTIW,
};

static const int defaultWindowSize = 60;
static const double defaultKParam  = 0.4;
static_assert(defaultWindowSize <= windowSizeLimit, "defaultWindowSize must not exceed windowSizeLimit.");

static ProgramMode programMode = OUT_BINARY;
static const char* filename_in;
static const char* filename_out;
static double preScale    = 1.0;
static int    windowSize  = defaultWindowSize;
static double kParam      = defaultKParam;
static double rScale      = 1.0;
static double tScale      = 1.0;
static double tBias       = 0.0;
static vector<int> multiWindowSize;



static void usage(int argc, char** argv, int ret = 1)
{
	fprintf(stderr,
		"usage: %s [-S SCALE] [-w WINDOW_SIZE] [-k K] [-r RSCALE] [-t T] [-T | -V | -X W1,W2,W3] IN OUT\n"
		"   -h | --help      show this help\n"
		"   -v | --version   show version information\n"
		"   -S SCALE         scale image by Lanczos4 prior to binarization [1.0]\n"
		"   -w WINDOW_SIZE   set window size          [%d]\n"
		"   -k K             set K parameter for Sauvola's algorithm [%f]\n"
		"   -r RSCALE        set scale of R parameter [1.0]\n"
		"                    (1.0 for maximum standard deviation possible)\n"
		"   -t T             set threshold scale      [1.0]\n"
		"   -b B             set threshold bias       [0.0]\n"
		"   -T               write threshold image instead of binary image\n"
		"   -V               write variable threshold image instead of standard image\n"
		"   -P               write pixelwise input image instead of binary image\n"
		"                    (RGB mapping: R=~intensity, G=variance, B=mean)\n"
		"   -X W1,W2,W3      write multi window size, variable threshold image\n"
		"                    (RGB mapping: R=W1, G=W2, B=W3)\n",
		argv[0], defaultWindowSize, defaultKParam);
	exit(ret);
}

static void argparse(int argc, char** argv)
{
	unordered_map<string, ProgramMode> pmodes = {
		{ "b",         OUT_BINARY },
		{ "binary",    OUT_BINARY },
		{ "binarized", OUT_BINARY },
		{ "t",         OUT_THRESHOLD },
		{ "threshold", OUT_THRESHOLD },
		{ "v",         OUT_VARIABLE },
		{ "variable",  OUT_VARIABLE },
		{ "p",         OUT_PIXELINFO },
		{ "pixels",    OUT_PIXELINFO },
		{ "pixelinfo", OUT_PIXELINFO },
		{ "multiw",          OUT_VARIABLE_MULTIW },
		{ "variable-multiw", OUT_VARIABLE_MULTIW },
	};
	const struct option longopts[] = {
		{ "help",              no_argument, 0, 'h' },
		{ "version",           no_argument, 0, 'v' },
		{ "prescale",          required_argument, 0, 'S' },
		{ "window-size",       required_argument, 0, 'w' },
		{ "k-param",           required_argument, 0, 'k' },
		{ "r-scale",           required_argument, 0, 'r' },
		{ "threshold-scale",   required_argument, 0, 't' },
		{ "threshold-bias",    required_argument, 0, 'b' },
		{ "output-type",       required_argument, 0, 'O' },
		{ "multi-window-size", required_argument, 0, 'X' },
		{},
	};
	int opt, longindex;
	try
	{
		opterr = 0;
		while ((opt = getopt_long(argc, argv, ":hvS:w:k:r:t:b:TVPX:", longopts, &longindex)) != -1)
		{
			switch (opt)
			{
				case 'h':
					usage(argc, argv, 0);
					break;
				case 'v':
					fprintf(stderr, "binarize-sauvola version %s\n%s\n", SOFTWARE_VERSION, SOFTWARE_COPYRIGHT);
					exit(0);
					break;
				case 'S':
					preScale = argparse_double("-S", optarg);
					if (preScale <= 0)
						throw argparse_error("-S", "prescale value must be positive.");
					break;
				case 'T':
					programMode = OUT_THRESHOLD;
					break;
				case 'V':
					programMode = OUT_VARIABLE;
					break;
				case 'P':
					programMode = OUT_PIXELINFO;
					break;
				case 'X':
				{
					programMode = OUT_VARIABLE_MULTIW;
					multiWindowSize.clear();
					string arg(optarg);
					size_t p0 = 0, p1;
					do
					{
						if (multiWindowSize.size() >= 3)
							throw argparse_error("-X", "too many window sizes.");
						p1 = min(arg.find_first_of(',', p0), arg.size());
						string token(arg, p0, p1 - p0);
						int wsize = argparse_int("-X", token.c_str());
						if (wsize < 1)
							throw argparse_error("-X", "one of the window sizes are too small.");
						if (wsize > windowSizeLimit)
							throw argparse_error("-X", "one of the window sizes are too large.");
						multiWindowSize.push_back(wsize);
						p0 = p1 + 1;
					} while (p1 < arg.size());
				}; break;
				case 'O':
				{
					auto p = pmodes.find(optarg);
					if (p == pmodes.end())
						throw argparse_error("--output-type", "unknown value.");
					programMode = p->second;
				}; break;
				case 'w':
					windowSize = argparse_int("-w", optarg);
					if (windowSize < 1)
						throw argparse_error("-w", "window size is too small.");
					if (windowSize > windowSizeLimit)
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
				case 't':
					tScale = argparse_double("-t", optarg);
					if (tScale <= 0)
						throw argparse_error("-t", "threshold scale must be larger than zero.");
					break;
				case 'b':
					tBias = argparse_double("-b", optarg);
					break;
				case ':':
					throw argparse_error(argv[0], "insufficient argument.");
					break;
				default: // '?'
					throw argparse_error(argv[0], "invalid option.");
					break;
			}
		}
		if (programMode == OUT_VARIABLE || programMode == OUT_VARIABLE_MULTIW)
		{
			if (rScale < 1)
				throw argparse_error("-r", "R scale must not be less than 1 if variable output is enabled.");
		}
		if (programMode == OUT_VARIABLE_MULTIW)
		{
			if (multiWindowSize.size() == 0)
				throw argparse_error("--output-type", "value of variable-multiw requires a `-X' option.");
			while (multiWindowSize.size() < 3)
				multiWindowSize.push_back(*(multiWindowSize.end() - 1));
			windowSize = *(max_element(multiWindowSize.begin(), multiWindowSize.end()));
		}
		else
		{
			multiWindowSize = { windowSize };
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



int main(int argc, char** argv)
{
	argparse(argc, argv);
	Mat img = imread(filename_in, IMREAD_GRAYSCALE);
	if (!img.data)
	{
		fprintf(stderr, "%s: image could not be loaded.\n", filename_in);
		return 1;
	}
	int w = img.cols;
	int h = img.rows;
	if (w == 0 || h == 0)
	{
		fprintf(stderr, "%s: image is empty.\n", filename_in);
		return 1;
	}

	// Resize image if necessary
	if (preScale != 1.0)
	{
		if (preScale * h + 1 >= numeric_limits<int>::max() || preScale * w + 1 >= numeric_limits<int>::max())
		{
			fprintf(stderr, "%s: image is too big after prescaling.\n", filename_in);
			return 1;
		}
		int nw = preScale * w;
		int nh = preScale * h;
		if (nw == 0 || nh == 0)
		{
			fprintf(stderr, "%s: image is empty after prescaling.\n", filename_in);
			return 1;
		}
		if (numeric_limits<int>::max() / nw < nh) // nw * nh > numeric_limits<int>::max()
		{
			fprintf(stderr, "%s: image is too big after prescaling.\n", filename_in);
			return 1;
		}
		if (w != nw || h != nh)
		{
			resize(img, img, Size(nw, nh), 0, 0, INTER_LANCZOS4);
			w = nw;
			h = nh;
		}
	}

	// Check padded image size
	if (
		numeric_limits<int>::max() - windowSize < w ||
		numeric_limits<int>::max() - windowSize < h ||
		numeric_limits<int>::max() / (w + windowSize) < h + windowSize
	)
	{
		fprintf(stderr, "%s: image size plus window size is too big to pad.\n", filename_in);
		return 1;
	}

	// Supplementary parameters
	double rParam = rScale * (255.0 * 0.5);
	double tRealBias = 255.0 * tBias;
	bool oVariable(programMode == OUT_VARIABLE || programMode == OUT_VARIABLE_MULTIW);

	// Iterate over window sizes
	Mat realdst;
	if (programMode == OUT_PIXELINFO || programMode == OUT_VARIABLE_MULTIW)
		realdst = Mat(h, w, CV_8UC3);
	for (size_t c = 0; c < multiWindowSize.size(); c++)
	{
		int wsize = multiWindowSize[c];
		int win_n = wsize / 2;
		int win_p = wsize / 2;
		if ((wsize % 2) != 0)
			++win_n;
		int pw = w + wsize;
		int ph = h + wsize;
		double invsqWindow = 1.0 / wsize / wsize;
		Mat dst(h, w, CV_8U);
		Mat padimg;
		copyMakeBorder(img, padimg, win_n, win_p, win_n, win_p, BORDER_REPLICATE);
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
			int Y1 = pw * (y + wsize);
			for (int x = 0; x < w; x++)
			{
				intimage_type total1 =
					buffer1[Y1 + (x + wsize)] -
					buffer1[Y1 + x] +
					buffer1[Y0 + x] -
					buffer1[Y0 + (x + wsize)];
				intimage_type total2 =
					buffer2[Y1 + (x + wsize)] -
					buffer2[Y1 + x] +
					buffer2[Y0 + x] -
					buffer2[Y0 + (x + wsize)];
				double mean   = total1 * invsqWindow;
				double stddev = sqrt(total2 * invsqWindow - mean * mean);
				if (programMode == OUT_PIXELINFO)
				{
					auto chI = 255 - padimg.at<unsigned char>(y + win_n, x + win_n);
					auto chD = stddev * 2.0;
					auto chM = mean;
					realdst.at<Vec3b>(y, x) = Vec3b(chM, chD, chI);
				}
				if (oVariable)
				{
					/*
						Variable Threshold Image:
						In Sauvola's algorithm, increasing K makes some black pixels white.
						The intensity of each pixel in this mode is determined by
						the lowest K value (Kt) which makes given pixel white.
						White: Kt == 0, Black: Kt >= 1
					*/
					double th1 = tScale * mean;
					double th0 = th1 * (1 + (stddev / rParam - 1));
					th0 += tRealBias; th1 += tRealBias;
					// th0 <= th1 while rScale >= 1.0.
					double v = padimg.at<unsigned char>(y + win_n, x + win_n);
					v = max(min(v, th1), th0);
					dst.at<unsigned char>(y, x) = 255.0 * (v - th0) / (th1 - th0);
				}
				else
				{
					int threshold = tScale * mean * (1 + kParam * (stddev / rParam - 1)) + tRealBias;
					if (programMode == OUT_THRESHOLD)
						dst.at<unsigned char>(y, x) = threshold;
					else
						dst.at<unsigned char>(y, x) = padimg.at<unsigned char>(y + win_n, x + win_n) > threshold ? 255 : 0;
				}
			}
		}
		// Finalization
		delete[] buffer1;
		delete[] buffer2;
		if (programMode == OUT_VARIABLE_MULTIW)
		{
			// Copy temporary image to a channel of output image
			size_t d = c + 1;
			if (d == multiWindowSize.size())
				d = 3;
			for (size_t k = c; k < d; k++)
			{
				size_t realch = 2 - k;
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						realdst.at<Vec3b>(y, x)[realch] = dst.at<unsigned char>(y, x);
			}
		}
		else if (programMode != OUT_PIXELINFO)
			realdst = dst;
	}
	vector<int> params;
	if (programMode == OUT_BINARY)
	{
		string fn(filename_out);
		if (fn.size() >= 4 && fn.substr(fn.size() - 4) == ".png")
		{
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(9);
			params.push_back(IMWRITE_PNG_BILEVEL);
		}
	}
	imwrite(filename_out, realdst, params);
	return 0;
}
