/*

	My Image Manipulation Tools for Digital Humanities
	Basic Binarization Testbed

	binarize.cpp

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

	This program is a wrapper of OpenCV's thresholding algorithms.

	It depends on:
	- C++11-compatible compiler
	- GNU-compatible getopt_long function.
	- OpenCV 3.x

*/

#define SOFTWARE_VERSION    "0.2.0"
#define SOFTWARE_COPYRIGHT  "Copyright (C) 2019 Tsukasa OI."

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <getopt.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "microlib/argparse.hpp"

using namespace std;
using namespace cv;



enum ProgramMode
{
	OUT_BINARIZE_CONST,
	OUT_BINARIZE_OTSU,
	OUT_BINARIZE_ADAPTIVE_MEAN,
	OUT_BINARIZE_ADAPTIVE_GAUSSIAN,
};

static const int defaultAdaptiveWindowSize = 3;
static_assert(defaultAdaptiveWindowSize > 1 && (defaultAdaptiveWindowSize % 2) == 1,
	"defaultAdaptiveWindowSize must be an odd number greater than 1.");

static ProgramMode programMode = OUT_BINARIZE_CONST;
static const char* filename_in;
static const char* filename_out = nullptr;
static double preScale       = 1.0;
static double constThreshold = 0.5;
static int    adaptiveWindowSize = defaultAdaptiveWindowSize;
static double adaptiveConst  = 0.0;



static void usage(int argc, char** argv, int ret = 1)
{
	fprintf(stderr,
		"usage: %s [-S SCALE] [-t THRESHOLD | -O | [-M | -G] [-w WINDOW_SIZE] [-c C]] IN [OUT]\n"
		"   -h | --help      show this help\n"
		"   -v | --version   show version information\n"
		"   -S SCALE         scale image by Lanczos4 prior to binarization [1.0]\n"
		"   -t THRESHOLD     set constant thresholding value\n"
		"   -O               perform Otsu's algorithm and write threshold value to stdout\n"
		"   -M               perform adaptive mean thresholding\n"
		"   -G               perform adaptive Gaussian thresholding\n"
		"   -w WINDOW_SIZE   set window size on adaptive thresholding   [%d]\n"
		"   -c C             set negative bias on adaptive thresholding [0.0]\n",
		argv[0], defaultAdaptiveWindowSize
	);
	exit(ret);
}

static void argparse(int argc, char** argv)
{
	unordered_map<string, ProgramMode> pmodes = {
		{ "b",                OUT_BINARIZE_CONST },
		{ "binarize",         OUT_BINARIZE_CONST },
		{ "binarize-static",  OUT_BINARIZE_CONST },
		{ "binarize-const",   OUT_BINARIZE_CONST },
		{ "threshold",        OUT_BINARIZE_CONST },
		{ "threshold-static", OUT_BINARIZE_CONST },
		{ "threshold-const",  OUT_BINARIZE_CONST },
		{ "adaptive-mean",   OUT_BINARIZE_ADAPTIVE_MEAN },
		{ "mean",            OUT_BINARIZE_ADAPTIVE_MEAN },
		{ "adaptive",          OUT_BINARIZE_ADAPTIVE_GAUSSIAN },
		{ "adaptive-gauss",    OUT_BINARIZE_ADAPTIVE_GAUSSIAN },
		{ "adaptive-gaussian", OUT_BINARIZE_ADAPTIVE_GAUSSIAN },
		{ "gauss",             OUT_BINARIZE_ADAPTIVE_GAUSSIAN },
		{ "gaussian",          OUT_BINARIZE_ADAPTIVE_GAUSSIAN },
		{ "otsu",          OUT_BINARIZE_OTSU },
		{ "get-threshold", OUT_BINARIZE_OTSU },
	};
	const struct option longopts[] = {
		{ "help",              no_argument, 0, 'h' },
		{ "version",           no_argument, 0, 'v' },
		{ "prescale",          required_argument, 0, 'S' },
		{ "threshold",         required_argument, 0, 't' },
		{ "mode",              required_argument, 0, 'm' },
		{ "window-size",       required_argument, 0, 'w' },
		{ "threshold-negbias", required_argument, 0, 'c' },
		{ "c-param",           required_argument, 0, 'c' },
		{},
	};
	int opt, longindex;
	try
	{
		opterr = 0;
		while ((opt = getopt_long(argc, argv, ":hvS:t:OMGw:c:", longopts, &longindex)) != -1)
		{
			switch (opt)
			{
				case 'h':
					usage(argc, argv, 0);
					break;
				case 'v':
					fprintf(stderr, "binarize version %s\n%s\n", SOFTWARE_VERSION, SOFTWARE_COPYRIGHT);
					exit(0);
					break;
				case 'S':
					preScale = argparse_double("-S", optarg);
					if (preScale <= 0)
						throw argparse_error("-S", "prescale value must be positive.");
					break;
				case 't':
					constThreshold = argparse_double("-t", optarg);
					if (constThreshold < 0)
						throw argparse_error("-t", "constant threshold must not be negative.");
					if (constThreshold > 1)
						throw argparse_error("-t", "constant threshold must not exceed one.");
					break;
				case 'O':
					programMode = OUT_BINARIZE_OTSU;
					break;
				case 'M':
					programMode = OUT_BINARIZE_ADAPTIVE_MEAN;
					break;
				case 'G':
					programMode = OUT_BINARIZE_ADAPTIVE_GAUSSIAN;
					break;
				case 'm':
				{
					auto p = pmodes.find(optarg);
					if (p == pmodes.end())
						throw argparse_error("--mode", "unknown value.");
					programMode = p->second;
				}; break;
				case 'w':
					adaptiveWindowSize = argparse_int("-w", optarg);
					if (adaptiveWindowSize <= 1)
						throw argparse_error("-w", "window size is too small.");
					if ((adaptiveWindowSize % 2) != 1)
						throw argparse_error("-w", "window size must be an odd number greater than one.");
					break;
				case 'c':
					adaptiveConst = argparse_double("-c", optarg);
					if (adaptiveConst < 0)
						throw argparse_error("-c", "C parameter must not be negative.");
					if (adaptiveConst > 1)
						throw argparse_error("-c", "C parameter must not exceed one.");
					break;
				case ':':
					throw argparse_error(argv[0], "insufficient argument.");
					break;
				default: // '?'
					throw argparse_error(argv[0], "invalid option.");
					break;
			}
		}
		if (argc - optind < 1 || argc - optind > 2)
			usage(argc, argv, 1);
		filename_in = argv[optind++];
		if (argc != optind)
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
	double realThreshold = constThreshold * 255.0;
	double realAdaptiveConst = adaptiveConst * 255.0;
	switch (programMode)
	{
		case OUT_BINARIZE_CONST:
			threshold(img, img, realThreshold, 255.0, THRESH_BINARY);
			break;
		case OUT_BINARIZE_OTSU:
			realThreshold = threshold(img, img, 0.0, 255.0, THRESH_OTSU | THRESH_BINARY);
			realThreshold /= 255.0;
			printf("%f\n", realThreshold);
			break;
		case OUT_BINARIZE_ADAPTIVE_MEAN:
		case OUT_BINARIZE_ADAPTIVE_GAUSSIAN:
			adaptiveThreshold(img, img, 255.0,
				programMode == OUT_BINARIZE_ADAPTIVE_MEAN ? ADAPTIVE_THRESH_MEAN_C : ADAPTIVE_THRESH_GAUSSIAN_C,
				THRESH_BINARY, adaptiveWindowSize, realAdaptiveConst);
			break;
	}
	if (!filename_out)
		return 0;
	vector<int> params;
	{
		string fn(filename_out);
		if (fn.size() >= 4 && fn.substr(fn.size() - 4) == ".png")
		{
			params.push_back(IMWRITE_PNG_COMPRESSION);
			params.push_back(9);
			params.push_back(IMWRITE_PNG_BILEVEL);
		}
	}
	imwrite(filename_out, img, params);
	return 0;
}
