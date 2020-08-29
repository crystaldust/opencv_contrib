// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __OPENCV_MCC_COLORSPACE_HPP__
#define __OPENCV_MCC_COLORSPACE_HPP__

#include <vector>
#include <string>
#include <iostream>
#include "opencv2/mcc/io.hpp"
#include "opencv2/mcc/operations.hpp"
#include "opencv2/mcc/utils.hpp"

namespace cv 
{
namespace ccm 
{

/**\Basic class for ColorSpace */
class ColorSpace 
{
public:
	IO io;
	std::string type;
	bool linear;
	Operations to;
	Operations from;
	ColorSpace* l = 0;
	ColorSpace* nl = 0;

	ColorSpace() {};

	ColorSpace(IO io, std::string type, bool linear) :io(io), type(type), linear(linear) {};

	virtual ~ColorSpace() 
	{
		l = 0;
		nl = 0;
	};

	virtual bool relate(const ColorSpace& other) const 
	{
		return (type == other.type) && (io == other.io);
	};

	virtual Operations relation(const ColorSpace& other) const 
	{
		return IDENTITY_OPS;
	};

	bool operator<(const ColorSpace& other)const 
	{
		return (io < other.io || (io == other.io && type < other.type) || (io == other.io && type == other.type && linear < other.linear));
	}
};

/* base of RGB color space;
	the argument values are from AdobeRGB;
	Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space */
class RGBBase_ : public ColorSpace 
{
public:
	//primaries
	double xr;
	double yr;
	double xg;
	double yg;
	double xb;
	double yb;
	MatFunc toL;
	MatFunc fromL;
	cv::Mat M_to;
	cv::Mat M_from;

	using ColorSpace::ColorSpace;

	/* There are 3 kinds of relationships for RGB:
		1. Different types;    - no operation
		1. Same type, same linear; - copy
		2. Same type, different linear, self is nonlinear; - 2 toL
		3. Same type, different linear, self is linear - 3 fromL*/
	virtual Operations relation(const ColorSpace& other) const 
	{
		if (linear == other.linear) 
		{ 
			return IDENTITY_OPS; 
		}
		if (linear) 
		{ 
			return Operations({ Operation(fromL) }); 
		}
		return Operations({ Operation(toL) });
	};

	void init() 
	{
		setParameter();
		calLinear();
		calM();
		calOperations();
	}

	/* produce color space instance with linear and non-linear versions */
	void bind(RGBBase_& rgbl) 
	{
		init();
		rgbl.init();
		l = &rgbl;
		rgbl.l = &rgbl;
		nl = this;
		rgbl.nl = this;
	}

private:
	virtual void setParameter() {};

	/* calculation of M_RGBL2XYZ_base;
		see ColorSpace.pdf for details; */
	virtual void calM() 
	{
		cv::Mat XYZr, XYZg, XYZb, XYZ_rgbl, Srgb;
		XYZr = cv::Mat(xyY2XYZ({ xr, yr }), true);
		XYZg = cv::Mat(xyY2XYZ({ xg, yg }), true);
		XYZb = cv::Mat(xyY2XYZ({ xb, yb }), true);
		merge(std::vector<cv::Mat>{ XYZr, XYZg, XYZb }, XYZ_rgbl);
		XYZ_rgbl = XYZ_rgbl.reshape(1, XYZ_rgbl.rows);
		cv::Mat XYZw = cv::Mat(illuminants.find(io)->second, true);
		solve(XYZ_rgbl, XYZw, Srgb);
		merge(std::vector<cv::Mat>{ Srgb.at<double>(0)* XYZr,
			Srgb.at<double>(1)* XYZg,
			Srgb.at<double>(2)* XYZb }, M_to);
		M_to = M_to.reshape(1, M_to.rows);
		M_from = M_to.inv();
	};

	/* operations to or from XYZ */
	virtual void calOperations() 
	{
		/* rgb -> rgbl */
		toL = [this](cv::Mat rgb)->cv::Mat {return toLFunc(rgb); };

		/* rgbl -> rgb */
		fromL = [this](cv::Mat rgbl)->cv::Mat {return fromLFunc(rgbl); };

		if (linear) 
		{
			to = Operations({ Operation(M_to.t()) });
			from = Operations({ Operation(M_from.t()) });
		}
		else 
		{
			to = Operations({ Operation(toL), Operation(M_to.t()) });
			from = Operations({ Operation(M_from.t()), Operation(fromL) });
		}
	}

	virtual void calLinear() {}

	virtual cv::Mat toLFunc(cv::Mat& rgb) 
	{ 
		return cv::Mat(); 
	};

	virtual cv::Mat fromLFunc(cv::Mat& rgbl) 
	{ 
		return cv::Mat(); 
	};

};

class AdobeRGBBase_ : public RGBBase_ 
{
public:
	using RGBBase_::RGBBase_;
	double gamma;

private:
	virtual cv::Mat toLFunc(cv::Mat& rgb) 
	{
		return gammaCorrection(rgb, gamma);
	}

	virtual cv::Mat fromLFunc(cv::Mat& rgbl) 
	{
		return gammaCorrection(rgbl, 1. / gamma);
	}
};

class sRGBBase_ : public RGBBase_ 
{
public:
	using RGBBase_::RGBBase_;
	double a;
	double gamma;
	double alpha;
	double beta;
	double phi;
	double K0;

private:
	/* linearization parameters
		see ColorSpace.pdf for details; */
	virtual void calLinear() 
	{
		alpha = a + 1;
		K0 = a / (gamma - 1);
		phi = (pow(alpha, gamma) * pow(gamma - 1, gamma - 1)) / (pow(a, gamma - 1) * pow(gamma, gamma));
		beta = K0 / phi;
	}

	double toLFuncEW(double& x) 
	{
		if (x > K0) 
		{
			return pow(((x + alpha - 1) / alpha), gamma);
		}
		else if (x >= -K0) 
		{
			return x / phi;
		}
		else 
		{
			return -(pow(((-x + alpha - 1) / alpha), gamma));
		}
	}

	/* linearization
		see ColorSpace.pdf for details; */
	cv::Mat toLFunc(cv::Mat& rgb) 
	{
		return elementWise(rgb, [this](double a)->double {return toLFuncEW(a); });
	}

	double fromLFuncEW(double& x) 
	{
		if (x > beta) 
		{
			return alpha * pow(x, 1 / gamma) - (alpha - 1);
		}
		else if (x >= -beta) 
		{
			return x * phi;
		}
		else {
			return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
		}
	}

	/* delinearization
		see ColorSpace.pdf for details; */
	cv::Mat fromLFunc(cv::Mat& rgbl) 
	{
		return elementWise(rgbl, [this](double a)->double {return fromLFuncEW(a); });
	}
};

class sRGB_ :public sRGBBase_ 
{
public:
	sRGB_(bool linear) :sRGBBase_(D65_2, "sRGB", linear) {};

private:
	/* base of sRGB-like color space;
		the argument values are from sRGB;
		data from https://en.wikipedia.org/wiki/SRGB */
	void setParameter() 
	{
		xr = 0.64;
		yr = 0.33;
		xg = 0.3;
		yg = 0.6;
		xb = 0.15;
		yb = 0.06;
		a = 0.055;
		gamma = 2.4;
	}
};

class AdobeRGB_ : public AdobeRGBBase_ 
{
public:
	AdobeRGB_(bool linear = false) :AdobeRGBBase_(D65_2, "AdobeRGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.64;
		yr = 0.33;
		xg = 0.21;
		yg = 0.71;
		xb = 0.15;
		yb = 0.06;
		gamma = 2.2;
	}
};

/* data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space */
class WideGamutRGB_ : public AdobeRGBBase_ 
{
public:
	WideGamutRGB_(bool linear = false) :AdobeRGBBase_(D50_2, "WideGamutRGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.7347;
		yr = 0.2653;
		xg = 0.1152;
		yg = 0.8264;
		xb = 0.1566;
		yb = 0.0177;
		gamma = 2.2;
	}
};

/* data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space */
class ProPhotoRGB_ : public AdobeRGBBase_ 
{
public:
	ProPhotoRGB_(bool linear = false) :AdobeRGBBase_(D50_2, "ProPhotoRGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.734699;
		yr = 0.265301;
		xg = 0.159597;
		yg = 0.840403;
		xb = 0.036598;
		yb = 0.000105;
		gamma = 1.8;
	}
};

/* data from https://en.wikipedia.org/wiki/DCI-P3 */
class DCI_P3_RGB_ : public AdobeRGBBase_ 
{
public:
	DCI_P3_RGB_(bool linear = false) :AdobeRGBBase_(D65_2, "DCI_P3_RGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.68;
		yr = 0.32;
		xg = 0.265;
		yg = 0.69;
		xb = 0.15;
		yb = 0.06;
		gamma = 2.2;
	}
};

/* data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html */
class AppleRGB_ : public AdobeRGBBase_ 
{
public:
	AppleRGB_(bool linear = false) :AdobeRGBBase_(D65_2, "AppleRGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.625;
		yr = 0.34;
		xg = 0.28;
		yg = 0.595;
		xb = 0.155;
		yb = 0.07;
		gamma = 1.8;
	}
};

/* data from https://en.wikipedia.org/wiki/Rec._709 */
class REC_709_RGB_ : public sRGBBase_ 
{
public:
	REC_709_RGB_(bool linear) :sRGBBase_(D65_2, "REC_709_RGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.64;
		yr = 0.33;
		xg = 0.3;
		yg = 0.6;
		xb = 0.15;
		yb = 0.06;
		a = 0.099;
		gamma = 1 / 0.45;
	}
};

/* data from https://en.wikipedia.org/wiki/Rec._2020 */
class REC_2020_RGB_ : public sRGBBase_ 
{
public:
	REC_2020_RGB_(bool linear) :sRGBBase_(D65_2, "REC_2020_RGB", linear) {};

private:
	void setParameter() 
	{
		xr = 0.708;
		yr = 0.292;
		xg = 0.17;
		yg = 0.797;
		xb = 0.131;
		yb = 0.046;
		a = 0.09929682680944;
		gamma = 1 / 0.45;
	}
};

sRGB_ sRGB(false), sRGBL(true);
AdobeRGB_ AdobeRGB(false), AdobeRGBL(true);
WideGamutRGB_ WideGamutRGB(false), WideGamutRGBL(true);
ProPhotoRGB_ ProPhotoRGB(false), ProPhotoRGBL(true);
DCI_P3_RGB_ DCI_P3_RGB(false), DCI_P3_RGBL(true);
AppleRGB_ AppleRGB(false), AppleRGBL(true);
REC_709_RGB_ REC_709_RGB(false), REC_709_RGBL(true);
REC_2020_RGB_ REC_2020_RGB(false), REC_2020_RGBL(true);

class ColorSpaceInitial 
{
public:
	ColorSpaceInitial() 
	{
		sRGB.bind(sRGBL);
		AdobeRGB.bind(AdobeRGBL);
		WideGamutRGB.bind(WideGamutRGBL);
		ProPhotoRGB.bind(ProPhotoRGBL);
		DCI_P3_RGB.bind(DCI_P3_RGBL);
		AppleRGB.bind(AppleRGBL);
		REC_709_RGB.bind(REC_709_RGBL);
		REC_2020_RGB.bind(REC_2020_RGBL);

	}
};

ColorSpaceInitial color_space_initial;

enum CAM 
{
	IDENTITY,
	VON_KRIES,
	BRADFORD
};

//todo 变量放在类内
static std::map <std::tuple<IO, IO, CAM>, cv::Mat > cams;
const static cv::Mat Von_Kries = (cv::Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081, -0.2263, 1.16532, 0.0457, 0., 0., 0.91822);
const static cv::Mat Bradford = (cv::Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
const static std::map <CAM, std::vector< cv::Mat >> MAs = {
	{IDENTITY , { cv::Mat::eye(cv::Size(3,3),CV_64FC1) , cv::Mat::eye(cv::Size(3,3),CV_64FC1)} },
	{VON_KRIES, { Von_Kries ,Von_Kries.inv() }},
	{BRADFORD, { Bradford ,Bradford.inv() }}
};

/* chromatic adaption matrices */
class XYZ :public ColorSpace 
{
public:
	XYZ(IO io) : ColorSpace(io, "XYZ", true) {};
	Operations cam(IO dio, CAM method = BRADFORD) 
	{
		return (io == dio) ? Operations() : Operations({ Operation(cam_(io, dio, method).t()) });
	}

private:
	/* get cam */
	cv::Mat cam_(IO sio, IO dio, CAM method = BRADFORD) const 
	{
		if (sio == dio) 
		{
			return cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
		}
		if (cams.count(std::make_tuple(dio, sio, method)) == 1) 
		{
			return cams[std::make_tuple(dio, sio, method)];
		}

		/* function from http ://www.brucelindbloom.com/index.html?ColorCheckerRGB.html */
		cv::Mat XYZws = cv::Mat(illuminants.find(dio)->second);
		cv::Mat XYZWd = cv::Mat(illuminants.find(sio)->second);
		cv::Mat MA = MAs.at(method)[0];
		cv::Mat MA_inv = MAs.at(method)[1];
		cv::Mat M = MA_inv * cv::Mat::diag((MA * XYZws) / (MA * XYZWd)) * MA;

		cams[std::make_tuple(dio, sio, method)] = M;
		cams[std::make_tuple(sio, dio, method)] = M.inv();
		return M;
	}

};

const XYZ XYZ_D65_2(D65_2);
const XYZ XYZ_D50_2(D50_2);

class Lab :public ColorSpace 
{
public:
	Lab(IO io) : ColorSpace(io, "XYZ", true) 
	{
		to = { Operation([this](cv::Mat src)->cv::Mat {return tosrc(src); }) };
		from = { Operation([this](cv::Mat src)->cv::Mat {return fromsrc(src); }) };
	}

private:
	static constexpr double delta = (6. / 29.);
	static constexpr double m = 1. / (3. * delta * delta);
	static constexpr double t0 = delta * delta * delta;
	static constexpr double c = 4. / 29.;

	cv::Vec3d fromxyz(cv::Vec3d& xyz) 
	{
		double x = xyz[0] / illuminants.find(io)->second[0], y = xyz[1] / illuminants.find(io)->second[1], z = xyz[2] / illuminants.find(io)->second[2];
		auto f = [this](double t)->double { return t > t0 ? std::cbrtl(t) : (m * t + c); };
		double fx = f(x), fy = f(y), fz = f(z);
		return { 116. * fy - 16. ,500 * (fx - fy),200 * (fy - fz) };
	}

	cv::Mat fromsrc(cv::Mat& src) 
	{
		return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return fromxyz(a); });
	}

	cv::Vec3d tolab(cv::Vec3d& lab) 
	{
		auto f_inv = [this](double t)->double {return t > delta ? pow(t, 3.0) : (t - c) / m; };
		double L = (lab[0] + 16.) / 116., a = lab[1] / 500., b = lab[2] / 200.;
		return { illuminants.find(io)->second[0] * f_inv(L + a),illuminants.find(io)->second[1] * f_inv(L),illuminants.find(io)->second[2] * f_inv(L - b) };
	}

	cv::Mat tosrc(cv::Mat& src) 
	{
		return channelWise(src, [this](cv::Vec3d a)->cv::Vec3d {return tolab(a); });
	}
};

const Lab Lab_D65_2(D65_2);
const Lab Lab_D50_2(D50_2);

} // namespace ccm
} // namespace cv


#endif