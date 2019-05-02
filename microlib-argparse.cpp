/*

	My Image Manipulation Tools for Digital Humanities
	Argument Parser Utility

	argparse.cpp

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

#include <cmath>

#include <stdexcept>
#include <string>

#include "microlib/argparse.hpp"



int argparse_int(const char* opt, const char* arg)
{
	try
	{
		size_t sz;
		int value = std::stoi(arg, &sz);
		if (arg[sz])
			throw std::invalid_argument("");
		return value;
	}
	catch (const std::invalid_argument&)
	{
		throw argparse_error(opt, "invalid argument.");
	}
	catch (const std::out_of_range&)
	{
		throw argparse_error(opt, "value out of range.");
	}
}

unsigned long argparse_ulong(const char* opt, const char* arg)
{
	try
	{
		size_t sz;
		unsigned long value = std::stoul(arg, &sz);
		if (arg[sz])
			throw std::invalid_argument("");
		return value;
	}
	catch (const std::invalid_argument&)
	{
		throw argparse_error(opt, "invalid argument.");
	}
	catch (const std::out_of_range&)
	{
		throw argparse_error(opt, "value out of range.");
	}
}

double argparse_double(const char* opt, const char* arg, bool allow_infinity, bool allow_nan)
{
	try
	{
		size_t sz;
		double value = std::stod(arg, &sz);
		if (arg[sz])
			throw std::invalid_argument("");
		if (!allow_nan && std::isnan(value))
			throw argparse_error(opt, "the value must not be NaN.");
		if (!allow_infinity && std::isinf(value))
			throw argparse_error(opt, "the value must not be infinity.");
		return value;
	}
	catch (const std::invalid_argument&)
	{
		throw argparse_error(opt, "invalid argument.");
	}
	catch (const std::out_of_range&)
	{
		throw argparse_error(opt, "value out of range.");
	}
}
