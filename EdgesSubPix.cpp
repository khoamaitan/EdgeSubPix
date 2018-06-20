#include <cmath>
#include <opencv2/opencv.hpp>
#include <algorithm>

#include "EdgesSubPix.h"

const double scale = 128.0;  // sum of half Canny filter is 128
const double sp::EdgesSubPix::UNDEFINED_RESPONSE = 9999.;
const double sp::EdgesSubPix::UNDEFINED_DIRECTION = 9999.;



sp::EdgesSubPix::EdgesSubPix()
{

}

sp::EdgesSubPix::~EdgesSubPix()
{

}

void getCannyKernel(cv::OutputArray _d, double alpha)
{
	// minmimum alpha has to be one
	if (alpha == 0)
	{
		alpha = 1;
	}

    int r = cvRound(alpha * 3);
    int ksize = 2 * r + 1;

    _d.create(ksize, 1, CV_16S, -1, true);

	cv::Mat k = _d.getMat();

    std::vector<float> kerF(ksize, 0.0f);
    kerF[r] = 0.0f;
    double a2 = alpha * alpha;
    float sum = 0.0f;
    for (int x = 1; x <= r; ++x)
    {
        float v = (float)(-x * std::exp(-x * x / (2 * a2)));
        sum += v;
        kerF[r + x] = v;
        kerF[r - x] = -v;
    }
    float scale = 128 / sum;
    for (int i = 0; i < ksize; ++i)
    {
        kerF[i] *= scale;
    }
	cv::Mat temp(ksize, 1, CV_32F, &kerF[0]);
    temp.convertTo(k, CV_16S);
}

// non-maximum supression and hysteresis
void postCannyFilter(const cv::Mat &src, cv::Mat &dx, cv::Mat &dy, int low, int high, cv::Mat &dst)
{
    ptrdiff_t mapstep = src.cols + 2;
	cv::AutoBuffer<uchar> buffer((src.cols + 2)*(src.rows + 2) + mapstep * 3 * sizeof(int));

    // L2Gradient comparison with square
    high = high * high;
    low = low * low;

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep;
    mag_buf[2] = mag_buf[1] + mapstep;
    memset(mag_buf[0], 0, mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
    (Top-Left Origin)

    1   2   3
    *  *  *
    * * *
    0*******0
    * * *
    *  *  *
    3   2   1
    */

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

#if CV_SSE2
    bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            int j = 0, width = src.cols;
#if CV_SSE2
            if (haveSSE2)
            {
                for (; j <= width - 8; j += 8)
                {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                    __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                    __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                    __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
            }
#elif CV_NEON
            for (; j <= width - 8; j += 8)
            {
                int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j, v_dst);

                v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j + 4, v_dst);
            }
#endif
            for (; j < width; ++j)
                _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];

            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm - 1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i - 1);
        const short* _y = dy.ptr<short>(i - 1);

        if ((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3 / 2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j - 1] && m >= _mag[j + 1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j + magstep2 - s] && m > _mag[j + magstep1 + s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
        __ocv_canny_push:
            if (!prev_flag && m > high && _map[j - mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep - 1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep + 1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep - 1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep + 1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}

double getAmplitude(cv::Mat &dx, cv::Mat &dy, int i, int j)
{
	cv::Point2d mag(dx.at<short>(i, j), dy.at<short>(i, j));
    return norm(mag);
}

void getMagNeighbourhood(cv::Mat &dx, cv::Mat &dy, cv::Point &p, int w, int h, std::vector<double> &mag)
{
    int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
    int down = p.y + 1 < h ? p.y + 1 : p.y;
    int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
    int right = p.x + 1 < w ? p.x + 1 : p.x;

    mag[0] = getAmplitude(dx, dy, top, left);
    mag[1] = getAmplitude(dx, dy, top, p.x);
    mag[2] = getAmplitude(dx, dy, top, right);
    mag[3] = getAmplitude(dx, dy, p.y, left);
    mag[4] = getAmplitude(dx, dy, p.y, p.x);
    mag[5] = getAmplitude(dx, dy, p.y, right);
    mag[6] = getAmplitude(dx, dy, down, left);
    mag[7] = getAmplitude(dx, dy, down, p.x);
    mag[8] = getAmplitude(dx, dy, down, right);
}

void get2ndFacetModelIn3x3(std::vector<double> &mag, std::vector<double> &a)
{
    a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] + 2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) / 9.0;
    a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
    a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
    a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] + mag[6] - 2.0 * mag[7] + mag[8]) / 6.0;
    a[4] = (-mag[0] + mag[2] 
		+ mag[6] - mag[8]) / 4.0;
    a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) + mag[6] + mag[7] + mag[8]) / 6.0;
}
/* 
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values. 
*/
void eigenvals(std::vector<double> &a, double eigval[2], double eigvec[2][2])
{
    // derivatives
    // fx = a[1], fy = a[2]
    // fxy = a[4]
    // fxx = 2 * a[3]
    // fyy = 2 * a[5]
    double dfdrc = a[4];
    double dfdcc = a[3] * 2.0;
    double dfdrr = a[5] * 2.0;
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5*(dfdcc - dfdrr) / dfdrc;
        t = 1.0 / (fabs(theta) + sqrt(theta*theta + 1.0));
        if (theta < 0.0) t = -t;
        c = 1.0 / sqrt(t*t + 1.0);
        s = t*c;
        e1 = dfdrr - t*dfdrc;
        e2 = dfdcc + t*dfdrc;
    }
    else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put that
    eigenvalue into first position.  If both are of equal absolute value, put
    the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    }
    else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    }
    else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        }
        else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

double vector2angle(double x, double y)
{
    double a = std::atan2(y, x);
    return a >= 0.0 ? a : a + CV_2PI;
}

void extractSubPixPoints(cv::Mat &dx, cv::Mat &dy, std::vector< sp::EdgesSubPix::Edge > &edgesInPixel, std::vector<sp::EdgesSubPix::Edge> &edges)
{
	int w = dx.cols;
	int h = dx.rows;
	edges.resize(edgesInPixel.size());

#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif

	for (int i = 0; i < (int)edgesInPixel.size(); ++i)
	{
		sp::EdgesSubPix::Edge &iedge = edgesInPixel[i];
		sp::EdgesSubPix::Edge &edge = edges[i];

		std::vector<double> magNeighbour(9);
		getMagNeighbourhood(dx, dy, (cv::Point)iedge.point, w, h, magNeighbour);
		std::vector<double> a(9);
		get2ndFacetModelIn3x3(magNeighbour, a);

		// Hessian eigen vector 
		double eigvec[2][2], eigval[2];
		eigenvals(a, eigval, eigvec);
		double t = 0.0;
		double ny = eigvec[0][0];
		double nx = eigvec[0][1];
		if (eigval[0] < 0.0)
		{
			double rx = a[1], ry = a[2], rxy = a[4], rxx = a[3] * 2.0, ryy = a[5] * 2.0;
			t = -(rx * nx + ry * ny) / (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny);
		}
		double px = nx * t;
		double py = ny * t;
		float x = (float)iedge.point.x;
		float y = (float)iedge.point.y;
		if (fabs(px) <= 0.5 && fabs(py) <= 0.5)
		{
            x += (float)(px+0.5);
            y += (float)(py+0.5);
		}
		edge.point = cv::Point2f(x, y);
		edge.response = (float)(a[0] / scale);
		edge.direction = (float)vector2angle(ny, nx);
        edge.nx = (float)nx;
        edge.ny = (float)ny;
	}
}

bool checkContourClosure(sp::EdgesSubPix::Contour& contour, bool displayReason=false)
{
	if (contour.points.size() < 3) {
		if (displayReason) std::cout << "contour < 3 pts" << std::endl;
		return false;
	}

	// compute occurences
	std::vector<cv::Point2f>::iterator maxX = std::max_element(contour.points.begin(), contour.points.end(), [&](const cv::Point2f& type, const cv::Point2f& next_type) { return type.x < next_type.x; });
	std::vector<cv::Point2f>::iterator maxY = std::max_element(contour.points.begin(), contour.points.end(), [&](const cv::Point2f& type, const cv::Point2f& next_type) { return type.y < next_type.y; });
	int w = maxX->x+1;
	int h = maxY->y+1;
	cv::Mat occurences(h, w, CV_8UC1, cv::Scalar(0));
	for (auto vec : contour.points) {

		int x = vec.x;
		int y = vec.y;
		occurences.at<uchar>(y, x)++;
	}
	/*if (!occurences.empty() & !occurences.empty()) {
		std::vector<int> compression;
		compression.push_back(CV_IMWRITE_PXM_BINARY);
		compression.push_back(0);
		cv::imwrite("occurences.pgm", occurences, compression);
	}*/

	std::vector<cv::Point> curve;
	cv::findNonZero(occurences, curve);
	
	// detect curve
	auto iter = std::find_if_not(curve.begin(), curve.end(), [occurences](cv::Point& pt) { return (occurences.at<uchar>(pt.y, pt.x) >= 2); });
	if (iter == curve.end())
	{
		return false;
	}

	//// colinearity check
	//bool colinear = true;
	//cv::Point2f vecOrig = contour.points[0];
	//std::vector<cv::Point2f> vecsOrigRef(contour.points.size() - 2, vecOrig);
	//cv::Point2f vecRef = contour.points[1] - contour.points[0];
	//std::vector<cv::Point2f> vecs;
	//std::transform(contour.points.begin() + 2, contour.points.end(), vecsOrigRef.begin(), std::back_inserter(vecs), std::minus<cv::Point2f>());

	//cv::Mat sys(2, 2, CV_64FC1);
	//for (auto vec : vecs) {
	//	sys.at<double>(0, 0) = vecRef.x; sys.at<double>(0, 1) = vec.x;
	//	sys.at<double>(1, 0) = vecRef.y; sys.at<double>(1, 1) = vec.y;
	//	double det = std::abs(cv::determinant(sys));
	//	if (det > 1e-3) {
	//		if (displayReason) std::cout << "determinant = " << det << std::endl;
	//		colinear = false;
	//		break;
	//	}
	//}
	//if (colinear) {
	//	if (displayReason) std::cout << "colinear" << std::endl;
	//	return false;
	//}

	//// closure check (points are ordered along the contour)
	//cv::Point2f closure = contour.points[contour.points.size() - 1] - contour.points[0];
	//double dist = cv::norm(closure);
	//if (displayReason) std::cout << "dist = " << dist << std::endl;
	//if (dist > 1)
	//{
	//	if (displayReason) std::cout << "closure > 1" << std::endl;
	//	return false;
	//}
	return true;
}

void extractSubPixPoints(cv::Mat &dx, cv::Mat &dy, std::vector<sp::EdgesSubPix::Contour> &contoursInPixel, std::vector<sp::EdgesSubPix::Contour> &contours)
{
    int w = dx.cols;
    int h = dx.rows;
	contours.resize(contoursInPixel.size());
    for (size_t i = 0; i < contoursInPixel.size(); ++i)
    {
		std::vector<cv::Point2f> &icontour = contoursInPixel[i].points;
		sp::EdgesSubPix::Contour &contour = contours[i];
        contour.points.resize(icontour.size());
        contour.response.resize(icontour.size());
        contour.direction.resize(icontour.size());
        contour.nx.resize(icontour.size());
        contour.ny.resize(icontour.size());
		contour.hierarchy = contoursInPixel[i].hierarchy;

#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)icontour.size(); ++j)
        {
			std::vector<double> magNeighbour(9);
            getMagNeighbourhood(dx, dy, (cv::Point)icontour[j], w, h, magNeighbour);
			std::vector<double> a(9);
            get2ndFacetModelIn3x3(magNeighbour, a);
           
            // Hessian eigen vector 
            double eigvec[2][2], eigval[2];
            eigenvals(a, eigval, eigvec);
            double t = 0.0;
            double ny = eigvec[0][0];
            double nx = eigvec[0][1];
            if (eigval[0] < 0.0)
            {
                double rx = a[1], ry = a[2], rxy = a[4], rxx = a[3] * 2.0, ryy = a[5] * 2.0;
                t = -(rx * nx + ry * ny) / (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny);
            }
            double px = nx * t;
            double py = ny * t;
            float x = (float)icontour[j].x;
            float y = (float)icontour[j].y;
            if (fabs(px) <= 0.5 && fabs(py) <= 0.5)
            { 
                x += (float)(px+0.5);
                y += (float)(py+0.5);
            }
            contour.points[j] = cv::Point2f(x, y);
            contour.response[j] = (float)(a[0] / scale);
            contour.direction[j] = (float)vector2angle(ny, nx);
            contour.nx[j] = (float)nx;
            contour.ny[j] = (float)ny;
        }

		std::vector<cv::Point2f> convexHullContour;
		cv::convexHull(icontour, convexHullContour);
		contour.length = cv::arcLength(convexHullContour, true);
		contour.area = cv::contourArea(convexHullContour);
    }
}

void extractPixPoints(std::vector<std::vector<cv::Point> > &contoursInPixel, std::vector<cv::Vec4i>& hierarchy, std::vector<sp::EdgesSubPix::Contour> &contours)
{
	contours.resize(contoursInPixel.size());
	for (size_t i = 0; i < contoursInPixel.size(); ++i)
	{
		std::vector<cv::Point> &icontour = contoursInPixel[i];
		sp::EdgesSubPix::Contour &contour = contours[i];
		contour.points.resize(icontour.size());
		contour.response.resize(icontour.size());
		contour.direction.resize(icontour.size());
        contour.nx.resize(icontour.size());
        contour.ny.resize(icontour.size());
		if (!hierarchy.empty()) {
			contour.hierarchy = hierarchy[i];
		}
		
#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif
		for (int j = 0; j < (int)icontour.size(); ++j)
		{
			
			float x = (float)icontour[j].x;
			float y = (float)icontour[j].y;

			contour.points[j] = cv::Point2f(x, y);
			contour.response[j] = sp::EdgesSubPix::UNDEFINED_RESPONSE;
			contour.direction[j] = sp::EdgesSubPix::UNDEFINED_DIRECTION;
            contour.nx[j] = sp::EdgesSubPix::UNDEFINED_DIRECTION;
            contour.ny[j] = sp::EdgesSubPix::UNDEFINED_DIRECTION;
		}

		std::vector<cv::Point> convexHullContour;
		cv::convexHull(icontour, convexHullContour);
		contour.length = cv::arcLength(convexHullContour, true);
		contour.area = cv::contourArea(convexHullContour);
	}
}

void extractPixPoints(const cv::Mat& edges, std::vector<sp::EdgesSubPix::Edge>& edgesInPixels)
{
	edgesInPixels.clear();
	for (int h = 0; h < edges.rows; h++)
	{
		const uchar * row = edges.ptr<uchar>(h);
		for (int w = 0; w < edges.cols; w++)
		{
			if (row[w] == 255) {
				sp::EdgesSubPix::Edge edge;
				edge.direction = sp::EdgesSubPix::UNDEFINED_DIRECTION;
				edge.point = cv::Point(w, h);
				edge.response = sp::EdgesSubPix::UNDEFINED_RESPONSE;
                edge.nx = sp::EdgesSubPix::UNDEFINED_RESPONSE;
                edge.ny = sp::EdgesSubPix::UNDEFINED_RESPONSE;
				edgesInPixels.push_back(edge);
			}
		}
	}
}

void roi2Image(const cv::Mat& gray, std::vector<sp::EdgesSubPix::Contour> &contours)
{
	// find ROI
	cv::Size wholeSize;
	cv::Point ofs;
	gray.locateROI(wholeSize, ofs);

	// adapt pt to ROI
	if (ofs != cv::Point(0, 0)) {
		for (size_t i = 0; i < contours.size(); ++i) {
			for (size_t j = 0; j < contours[i].points.size(); ++j) {
				contours[i].points[j].x += ofs.x;
				contours[i].points[j].y += ofs.y;
			}
		}
	}
}

void roi2Image(const cv::Mat& gray, std::vector<sp::EdgesSubPix::Edge> &edges)
{
	// find ROI
	cv::Size wholeSize;
	cv::Point ofs;
	gray.locateROI(wholeSize, ofs);

	// adapt pt to ROI
	if (ofs != cv::Point(0, 0)) {
		for (size_t i = 0; i < edges.size(); ++i) {
			edges[i].point.x += ofs.x;
			edges[i].point.y += ofs.y;
		}
	}
}

void roi2Image(const cv::Mat& gray, std::vector<cv::Point2f> &edges)
{
	// find ROI
	cv::Size wholeSize;
	cv::Point ofs;
	gray.locateROI(wholeSize, ofs);

	// adapt pt to ROI
	if (ofs != cv::Point(0, 0)) {
		for (size_t i = 0; i < edges.size(); ++i) {
			edges[i].x += ofs.x;
			edges[i].y += ofs.y;
		}
	}
}

void edgesPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<sp::EdgesSubPix::Edge>& edgesInPixel, cv::Mat& edges, cv::Mat& dx, cv::Mat& dy)
{
	// smooth edges noise
	cv::Mat blur;

	if (alpha > 0) {
		cv::GaussianBlur(gray, blur, cv::Size(0, 0), alpha, alpha);
	}
	else {
		blur = gray;
	}

	// gradient images
	cv::Mat d;
	getCannyKernel(d, alpha);
	cv::Mat one = cv::Mat::ones(cv::Size(1, 1), CV_16S);
	cv::sepFilter2D(blur, dx, CV_16S, d, one);
	cv::sepFilter2D(blur, dy, CV_16S, one, d);

	// non-maximum supression & hysteresis threshold
	if (edges.empty()) {
		edges = cv::Mat::zeros(gray.size(), CV_8UC1);
	}
	int lowThresh = cvRound(scale * low);
	int highThresh = cvRound(scale * high);
	postCannyFilter(gray, dx, dy, lowThresh, highThresh, edges);
	
	// apply mask
	if (!mask.empty()) {
		cv::bitwise_and(edges, mask, edges);
	}

	// format edges
	extractPixPoints(edges, edgesInPixel);
}

void contoursPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector < sp::EdgesSubPix::Edge > & edgesInPixel, std::vector<sp::EdgesSubPix::Contour>& contoursPix, int mode, cv::Mat& dx, cv::Mat& dy, cv::Mat& edges)
{
	// extract edges
	edgesPix(gray, alpha, low, high, mask, edgesInPixel, edges, dx, dy);

	// extract contours
	std::vector<std::vector<cv::Point> > contoursInPixel;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contoursInPixel, hierarchy, mode, cv::CHAIN_APPROX_NONE);

	// format contours
	extractPixPoints(contoursInPixel, hierarchy, contoursPix);
}

//---------------------------------------------------------------------
//          INTERFACE FUNCTIONS
//---------------------------------------------------------------------

void sp::EdgesSubPix::edgesSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<sp::EdgesSubPix::Edge>& edgesInPixel, std::vector<sp::EdgesSubPix::Edge>& edgesInSubPixel, cv::Mat& edges)
{
	// extract edges
	cv::Mat dx;
	cv::Mat dy;
	edgesPix(gray, alpha, low, high, mask, edgesInPixel, edges, dx, dy);

	// subpixel position extraction with steger's method and facet model 2nd polynominal in 3x3 neighbourhood
	extractSubPixPoints(dx, dy, edgesInPixel, edgesInSubPixel);

	// roi to gray
	roi2Image(gray, edgesInPixel);
	roi2Image(gray, edgesInSubPixel);
}

void sp::EdgesSubPix::contoursSubPix(const cv::Mat &gray, double alpha, int low, int high, const cv::Mat& mask, std::vector<sp::EdgesSubPix::Edge>& edgesInPixel, std::vector<sp::EdgesSubPix::Contour>& contoursInPixel, std::vector<sp::EdgesSubPix::Contour> &contoursInSubPixel, int mode, cv::Mat& edges)
{
	// extract contours
	cv::Mat dx;
	cv::Mat dy;
	contoursPix(gray, alpha, low, high, mask, edgesInPixel, contoursInPixel, mode, dx, dy, edges);

	// subpixel position extraction with steger's method and facet model 2nd polynominal in 3x3 neighbourhood
	extractSubPixPoints(dx, dy, contoursInPixel, contoursInSubPixel);

	// roi to gray
	roi2Image(gray, edgesInPixel);
	roi2Image(gray, contoursInPixel);
	roi2Image(gray, contoursInSubPixel);
}
