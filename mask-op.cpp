/*

	My Image Manipulation Tools for Digital Humanities
	Basic Mask Operation Testbed

	mask-op.cpp

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

	This program handles various masks (white inside and black outside).

	It depends on:
	- C++11-compatible compiler
	- GNU-compatible getopt_long function.
	- OpenCV 3.x

*/

#define SOFTWARE_VERSION    "0.3.0"
#define SOFTWARE_COPYRIGHT  "Copyright (C) 2019 Tsukasa OI."

#include <cmath>
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



enum ProgramOp
{
	MASK_CMD_NEG,
	MASK_CMD_FILL_BORDER,
	MASK_CMD_INSET_L2,
	MASK_CMD_INSET_L1,
};

struct ProgramCommand
{
	ProgramOp op;
	double dist;
};

static vector<ProgramCommand> commands;
static const char* filename_in;
static const char* filename_out;



static void usage(int argc, char** argv, int ret = 1)
{
	fprintf(stderr,
		"usage: %s [COMNANDS...] IN OUT\n"
		"   -h | --help          show this help\n"
		"   -v | --version       show version information\n"
		"COMMANDS:\n"
		"   -n | --neg           negate mask\n"
		"   -B | --border-fill   fill border with black\n"
		"   -i | --inset    WIDTH   shrink mask by WIDTH\n"
		"   -I | --inset-L1 WIDTH   (do the same but with L1 norm)\n"
		"   -o | --outset    WIDTH  grow mask by WIDTH\n"
		"   -O | --outset-L1 WIDTH  (do the same but with L1 norm)\n",
		argv[0]
	);
	exit(ret);
}

static void argparse(int argc, char** argv)
{
	const struct option longopts[] = {
		{ "help",            no_argument, 0, 'h' },
		{ "version",         no_argument, 0, 'v' },
		{ "neg",             no_argument, 0, 'n' },
		{ "border-fill",     no_argument, 0, 'B' },
		{ "inset",     required_argument, 0, 'i' },
		{ "inset-L1",  required_argument, 0, 'I' },
		{ "outset",    required_argument, 0, 'o' },
		{ "outset-L1", required_argument, 0, 'O' },
		{},
	};
	int opt, longindex;
	try
	{
		commands.clear();
		opterr = 0;
		while ((opt = getopt_long(argc, argv, ":hvnBi:I:o:O:", longopts, &longindex)) != -1)
		{
			switch (opt)
			{
				case 'h':
					usage(argc, argv, 0);
					break;
				case 'v':
					fprintf(stderr, "mask-op version %s\n%s\n", SOFTWARE_VERSION, SOFTWARE_COPYRIGHT);
					exit(0);
					break;
				case 'n':
					commands.push_back( { MASK_CMD_NEG } );
					break;
				case 'B':
					commands.push_back( { MASK_CMD_FILL_BORDER } );
					break;
				case 'i':
				case 'I':
				case 'o':
				case 'O':
				{
					string arg("-");
					arg.push_back(opt);
					double width = argparse_double(arg.c_str(), optarg);
					char realopt = opt;
					if (width < 0)
					{
						switch (opt)
						{
							case 'i':  realopt = 'o'; break;
							case 'I':  realopt = 'O'; break;
							case 'o':  realopt = 'i'; break;
							case 'O':  realopt = 'I'; break;
						}
						width = -width;
					}
					switch (realopt)
					{
						case 'i':
							commands.push_back( { MASK_CMD_INSET_L2, width } );
							break;
						case 'I':
							commands.push_back( { MASK_CMD_INSET_L1, width } );
							break;
						case 'o':
							// outset command is performed as negative inset
							commands.push_back( { MASK_CMD_NEG } );
							commands.push_back( { MASK_CMD_INSET_L2, width } );
							commands.push_back( { MASK_CMD_NEG } );
							break;
						case 'O':
							// outset command is performed as negative inset
							commands.push_back( { MASK_CMD_NEG } );
							commands.push_back( { MASK_CMD_INSET_L1, width } );
							commands.push_back( { MASK_CMD_NEG } );
							break;
					}
				}; break;
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



int main(int argc, char** argv)
{
	argparse(argc, argv);
	Mat img = imread(filename_in, IMREAD_GRAYSCALE);
	if (!img.data)
	{
		fprintf(stderr, "%s: image could not be loaded.\n", filename_in);
		return 1;
	}
	Mat tmp;
	int w = img.cols;
	int h = img.rows;
	for (auto cmd : commands)
	{
		switch (cmd.op)
		{
			case MASK_CMD_NEG:
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						img.at<unsigned char>(y, x) = 255 - img.at<unsigned char>(y, x);
				break;
			case MASK_CMD_FILL_BORDER:
				for (int x = 0; x < w; x++)
				{
					if (img.at<unsigned char>(0, x))
						floodFill(img, Point(x, 0), Scalar(0));
					if (img.at<unsigned char>(h-1, x))
						floodFill(img, Point(x, h-1), Scalar(0));
				}
				for (int y = 0; y < h; y++)
				{
					if (img.at<unsigned char>(y, 0))
						floodFill(img, Point(0, y), Scalar(0));
					if (img.at<unsigned char>(y, w-1))
						floodFill(img, Point(w-1, y), Scalar(0));
				}
				break;
			case MASK_CMD_INSET_L2:
				distanceTransform(img, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						img.at<unsigned char>(y, x) = (tmp.at<float>(y, x) <= cmd.dist) ? 0 : 255;
				break;
			case MASK_CMD_INSET_L1:
				distanceTransform(img, tmp, CV_DIST_L1, CV_DIST_MASK_PRECISE);
				for (int y = 0; y < h; y++)
					for (int x = 0; x < w; x++)
						img.at<unsigned char>(y, x) = (tmp.at<float>(y, x) <= cmd.dist) ? 0 : 255;
				break;
		}
	}
	vector<int> params;
	{
		string fn(filename_out);
		if (fn.size() >= 4 && fn.substr(fn.size() - 4) == ".png")
			params.push_back(IMWRITE_PNG_BILEVEL);
	}
	imwrite(filename_out, img, params);
	return 0;
}
